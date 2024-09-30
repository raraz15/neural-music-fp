"""Train a neural audio fingerprinter."""

import argparse

from typing import Tuple

import tensorflow as tf

from nmfp.dataset_dev import DevelopmentDataset
from nmfp.model.utils import build_fp
from nmfp.model.NTxent_loss import NTxentLoss
from nmfp.model.nnfp import FingerPrinter
from nmfp.model.specaug_chain.specaug_chain import SpecAugChainer
from nmfp.experiment_helper import ExperimentHelper
from nmfp.utils import load_config, print_config, set_seed
from nmfp.gpu import choose_first_gpu, set_global_determinism


@tf.function
def train_step(
    X: Tuple,
    m_specaug: SpecAugChainer,
    m_fp: FingerPrinter,
    loss_obj: NTxentLoss,
    helper: ExperimentHelper,
) -> float:
    """Perform a training step: forward, loss computation, and backpropagation.

    Parameters
    ----------
        X: (Xa, Xp) tuple of tensors.
            Xa: anchors (originals), s.t. [xa_0, xa_1,...]
            Xp: degraded replicas, s.t. [xp_0, xp_1] with xp_n = rand_aug(xa_n).
        m_specaug: SpecAugChainer object.
        m_fp: FingerPrinter object (model).
        loss_obj: NTxentLoss object.
        helper: ExperimentHelper object.

    Returns
    -------
        avg_loss: The cumulative average loss until current
            step within the current epoch.
    """

    assert len(X) == 2, "X must be a tuple of two elements."
    assert len(X[0]) == len(X[1]), "Xa and Xp must have the same length."

    # Number of anchors=positives
    n_anchors = len(X[0])

    # Concatenate the anchors and positives
    X = tf.concat(X, axis=0)

    # Apply the spec-augment chain
    feat = m_specaug(X)  # (nA+nP, F, T, 1)

    # Forward Pass
    m_fp.trainable = True
    with tf.GradientTape() as t:
        emb = m_fp(feat)  # (BSZ, Dim)
        # Calculate the loss
        loss, _, _ = loss_obj.compute_loss(
            emb[:n_anchors, :], emb[n_anchors:, :]
        )  # {emb_org, emb_rep}
        if m_fp.mixed_precision:
            scaled_loss = helper.optimizer.get_scaled_loss(loss)

    # Backward Pass
    if m_fp.mixed_precision:
        scaled_g = t.gradient(scaled_loss, m_fp.trainable_variables)
        g = helper.optimizer.get_unscaled_gradients(scaled_g)
    else:
        g = t.gradient(loss, m_fp.trainable_variables)
    helper.optimizer.apply_gradients(zip(g, m_fp.trainable_variables))

    # Update the epoch loss
    avg_loss = helper.update_tr_loss(loss)  # To tensorboard

    return avg_loss


def main(cfg: dict, cpu_n_workers: int, cpu_max_que: int, reduce_tracks: float) -> None:
    """Main training function."""

    # Get the dataloader
    print("-----------Initializing the datasets-----------")
    dataset = DevelopmentDataset(cfg)
    train_loader = dataset.get_train_loader(reduce_tracks)

    # Build the model
    print("-----------Building the model-----------")
    m_specaug, m_fp = build_fp(cfg)

    # Learning schedule
    total_nsteps = cfg["TRAIN"]["MAX_EPOCH"] * len(train_loader)
    if cfg["TRAIN"]["LR"]["SCHEDULE"].upper() == "COS":
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=float(cfg["TRAIN"]["LR"]["INITIAL_RATE"]),
            decay_steps=total_nsteps,
            alpha=float(cfg["TRAIN"]["LR"]["ALPHA"]),
        )
    elif cfg["TRAIN"]["LR"]["SCHEDULE"].upper() == "COS-RESTART":
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=float(cfg["TRAIN"]["LR"]["INITIAL_RATE"]),
            first_decay_steps=int(total_nsteps * 0.1),
            # num_periods=0.5, # doesnt exist in current tensorflow
            alpha=float(cfg["TRAIN"]["LR"]["ALPHA"]),
        )  # Default 2e-6
    else:
        lr_schedule = float(cfg["TRAIN"]["LR"]["INITIAL_RATE"])

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    if cfg["TRAIN"]["MIXED_PRECISION"]:
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    # Experiment helper: see utils.experiment_helper.py for details.
    helper = ExperimentHelper(
        cfg,
        optimizer=opt,
        model_to_checkpoint=m_fp,
    )

    # Loss object
    loss_obj_train = NTxentLoss(
        n_org=train_loader.n_anchor,
        n_rep=train_loader.n_pos,
        tau=cfg["TRAIN"]["LOSS"]["TAU"],
    )

    # Training loop
    ep_start = helper.epoch
    ep_max = cfg["TRAIN"]["MAX_EPOCH"]
    if ep_start != 1:
        assert ep_start <= ep_max, (
            f"When continuing training, MAX_EPOCH={ep_max} "
            f"must be greater than or equal to where training was left off, which is {ep_start}"
        )
    print("-----------Training starts-----------")
    for ep in range(ep_start, ep_max + 1):
        print(f"Epoch: {ep}/{ep_max}")

        """ Parallelism to speed up preprocessing.............. """
        enq = tf.keras.utils.OrderedEnqueuer(
            train_loader,
            use_multiprocessing=True,
            # We shuffle inside the dataset
            # OrderedEnqueuer calls train_loader.on_epoch_end()
            shuffle=False,
        )
        enq.start(workers=cpu_n_workers, max_queue_size=cpu_max_que)
        i = 0
        while i < len(enq.sequence):
            _, _, Xa, Xp = next(enq.get())
            avg_loss = train_step((Xa, Xp), m_specaug, m_fp, loss_obj_train, helper)
            i += 1
            if i + 1 == 1 or (i + 1) % 100 == 0 or i + 1 == len(enq.sequence):
                print(f"Step {i+1}/{len(enq.sequence)}: tr_loss:{avg_loss:.4f}")
        enq.stop()
        """ End of Parallelism................................. """

        # On epoch end
        print(f"Average train loss of the epoch:{helper._tr_loss.result():.4f}")
        helper.update_on_epoch_end(save_checkpoint_now=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the model and training configuation file.",
    )
    parser.add_argument(
        "--max_epoch",
        default=None,
        type=int,
        help="Maximum epochs to train. By default uses the value in the config file. "
        "Can be used to continue training. If provided it will override the config file.",
    )
    parser.add_argument(
        "--cpu_n_workers",
        default=10,
        type=int,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--cpu_max_que", default=20, type=int, help="Max queue size for data loading."
    )
    parser.add_argument(
        "--deterministic",
        default=False,
        action="store_true",
        help="Set the CUDA operaitions to be deterministic for the price of slow computations.",
    )
    parser.add_argument(
        "--reduce_tracks",
        default=100,
        type=float,
        help="Reduce training tracks size to this percentage. 100 uses all the tracks.",
    )
    args = parser.parse_args()

    # Training settings
    set_seed()
    choose_first_gpu()
    if args.deterministic:
        set_global_determinism()

    # Load the config file
    cfg = load_config(args.config_path)

    # Update the config file
    if args.max_epoch is not None:
        cfg["TRAIN"]["MAX_EPOCH"] = args.max_epoch

    # Print the config file
    print_config(cfg)

    # Train
    main(cfg, args.cpu_n_workers, args.cpu_max_que, args.reduce_tracks)

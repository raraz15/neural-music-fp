"""Train a neural music fingerprinter."""

import argparse
import gc
from typing import Tuple, Union

import tensorflow as tf

from nmfp.dataloaders import DevelopmentDataset
from nmfp.model.utils import build_fp
from nmfp.losses import (
    AlignmentUniformityLoss,
    DCL,
    KCL,
    MultiPosConLoss,
    TripletLoss,
)
from nmfp.model.nnfp import FingerPrinter
from nmfp.model.specaug_chain.specaug_chain import SpecAugChainer
from nmfp.experiment_helper import ExperimentHelper
from nmfp.utils import load_config, print_config, set_seed
from nmfp.gpu import set_global_determinism, set_gpu_memory_growth


@tf.function
def train_step(
    X: Tuple,
    m_specaug: SpecAugChainer,
    m_fp: FingerPrinter,
    loss_obj: Union[AlignmentUniformityLoss, DCL, KCL, MultiPosConLoss, TripletLoss],
    helper: ExperimentHelper,
) -> float:
    """
    Parameters
    ----------
        X: tensor (n_a, 1+n_ppa, n_mels, T)
        m_specaug: SpecAugChainer object.
        m_fp: FingerPrinter object (model).
        loss_obj: Loss object.
        helper: ExperimentHelper object.

    Returns
    -------
        avg_loss: The cumulative average loss until current
            step within the current epoch.
    """

    assert len(tf.shape(X)) == 4, "X must be a 4D tensor."

    n_anchors = tf.shape(X)[0]
    n_views = tf.shape(X)[1]  # 1 + n_ppa

    # Concatenate the anchors and positives, add channel dimension
    X = tf.reshape(X, (n_anchors * n_views, X.shape[2], X.shape[3], 1))

    # Apply the spec-augment chain
    feat = m_specaug(X)  # (n_anchors * n_views, F, T, 1)

    # Forward Pass
    m_fp.trainable = True
    with tf.GradientTape() as t:
        emb = m_fp(feat)  # (n_anchors * n_views, d)
        emb = tf.reshape(emb, (n_anchors, n_views, -1))
        loss = loss_obj(emb)
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

    try:

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
        if cfg["TRAIN"]["LOSS"]["LOSS_MODE"].upper() in ["NTXENT", "MULTIPOSCON"]:
            loss_obj_train = MultiPosConLoss(
                tau=cfg["TRAIN"]["LOSS"]["TAU"],
            )
        elif cfg["TRAIN"]["LOSS"]["LOSS_MODE"].upper() == "TRIPLET":
            loss_obj_train = TripletLoss(
                margin=cfg["TRAIN"]["LOSS"]["MARGIN"],
                pos_mode=cfg["TRAIN"]["LOSS"]["POS_MODE"],
                neg_mode=cfg["TRAIN"]["LOSS"]["NEG_MODE"],
                squared=cfg["TRAIN"]["LOSS"]["SQUARED"],
            )
        elif cfg["TRAIN"]["LOSS"]["LOSS_MODE"].upper() == "ALIGNMENT_UNIFORMITY":
            loss_obj_train = AlignmentUniformityLoss(
                alpha=cfg["TRAIN"]["LOSS"]["ALPHA"],
                t=cfg["TRAIN"]["LOSS"]["T"],
                w_alignment=cfg["TRAIN"]["LOSS"]["W_ALIGNMENT"],
                w_uniformity=cfg["TRAIN"]["LOSS"]["W_UNIFORMITY"],
            )
        elif cfg["TRAIN"]["LOSS"]["LOSS_MODE"].upper() == "DCL":
            loss_obj_train = DCL(
                tau=cfg["TRAIN"]["LOSS"]["TAU"],
            )
        elif cfg["TRAIN"]["LOSS"]["LOSS_MODE"].upper() == "KCL":
            loss_obj_train = KCL(
                t=cfg["TRAIN"]["LOSS"]["T"],
                gamma=cfg["TRAIN"]["LOSS"]["GAMMA"],
            )
        else:
            raise ValueError(
                f"Unknown loss LOSS_MODE: {cfg['TRAIN']['LOSS']['LOSS_MODE']}. "
                "Expected one of ['NTxent', 'Alignment_Uniformity']"
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

            progbar = tf.keras.utils.Progbar(
                len(train_loader), stateful_metrics=["tr loss"]
            )
            """ Parallelism to speed up preprocessing.............. """
            enq = tf.keras.utils.OrderedEnqueuer(
                train_loader,
                use_multiprocessing=True,
                shuffle=False,  # Already shuffled in dataset
            )
            enq.start(workers=cpu_n_workers, max_queue_size=cpu_max_que)
            gen = enq.get()  # Get the generator only once
            i = 0
            while i < len(enq.sequence):
                _, X = next(gen)
                avg_loss = train_step(X, m_specaug, m_fp, loss_obj_train, helper)
                i += 1
                progbar.add(1, values=[("tr loss", avg_loss)])
            enq.stop()
            """ End of Parallelism................................. """

            # On epoch end
            print(f"Average train loss of the epoch:{helper._tr_loss.result():.4f}")
            helper.update_on_epoch_end(save_checkpoint_now=True)

    finally:
        print("-----------Training ends-----------")
        tf.keras.backend.clear_session()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the configuation file. Examples in `config/` directory.",
    )
    parser.add_argument(
        "--workers",
        "-w",
        default=20,
        type=int,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--queue", "-q", default=40, type=int, help="Max queue size for data loading."
    )
    parser.add_argument(
        "--block-growth",
        default=False,
        action="store_true",
        help="Block GPU memory growth. I would allow growth. It speeds up considerably.",
    )
    parser.add_argument(
        "--deterministic",
        default=False,
        action="store_true",
        help="Set the CUDA operaitions to be deterministic for the price of slow computations.",
    )
    parser.add_argument(
        "--max-epoch",
        default=None,
        type=int,
        help="Maximum epochs to train. By default uses the value in the config file. "
        "Can be used to continue training. If provided it will override the config file.",
    )
    parser.add_argument(
        "--reduce-tracks",
        default=100,
        type=float,
        help="""Reduce training tracks size to this percentage. Can be used for debugging. 
        100 uses all the tracks.""",
    )
    args = parser.parse_args()

    set_seed()

    # Training settings
    tf.keras.backend.clear_session()

    if not args.block_growth:
        print("GPU memory growth is allowed.")
        set_gpu_memory_growth()

    if args.deterministic:
        set_global_determinism()

    # Load the config file
    cfg = load_config(args.config_path)

    # Update the config file
    if args.max_epoch is not None:
        cfg["MODEL"]["MAX_EPOCH"] = args.max_epoch

    # Print the config file
    print_config(cfg)

    # Train
    main(cfg, args.workers, args.queue, args.reduce_tracks)

    tf.keras.backend.clear_session()
    gc.collect()

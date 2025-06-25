import os

from typing import Tuple

import tensorflow as tf

from nmfp.model.nnfp import FingerPrinter
from nmfp.model.specaug_chain.specaug_chain import (
    get_specaug_chain_layer,
    SpecAugChainer,
)


def build_fp(cfg: dict) -> Tuple[SpecAugChainer, FingerPrinter]:
    """Build a FingerPrinter model and the spec-augmentation layer.

    Parameters
    ----------
        cfg : dict
            Configuration dictionary created from the '.yaml' located in config/ directory.

    Returns
    -------
        m_specaug : SpecAugChainer
            A SpecAugChainer object configured according to the provided configuration.
        m_fp : FingerPrinter
            A FingerPrinter object configured according to the provided configuration.

    """

    # Spec-augmentation layer.
    m_specaug = get_specaug_chain_layer(cfg, trainable=False)
    assert m_specaug.bypass == False  # Detachable by setting m_specaug.bypass.

    # Fingerprinter
    m_fp = get_fingerprinter(cfg, trainable=False)

    return m_specaug, m_fp


def get_fingerprinter(cfg: dict, trainable: bool = False) -> FingerPrinter:
    """Create a FingerPrinter object.

    Parameters
    ----------
        cfg : dict
            Configuration dictionary created from the '.yaml' located in config/ directory.
        trainable : bool, optional
            If True, the returned FingerPrinter object will be trainable.

    Returns
    -------
        m_fp : FingerPrinter
            A FingerPrinter object configured according to the provided configuration.

    """

    # Enable mixed precision after creating the processing layers.
    if cfg["TRAIN"]["MIXED_PRECISION"]:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled.")

    m_fp = FingerPrinter(
        emb_sz=cfg["MODEL"]["ARCHITECTURE"]["EMB_SZ"],
        fc_unit_dim=[32, 1],
        norm=cfg["MODEL"]["ARCHITECTURE"]["BN"],
        mixed_precision=cfg["TRAIN"]["MIXED_PRECISION"],
    )
    m_fp.trainable = trainable

    return m_fp


def get_checkpoint_index_and_restore_model(
    m_fp: FingerPrinter, checkpoint_dir: str, checkpoint_index: int = 0
) -> int:
    """Load the weights of a trained fingerprinter to an initialized model
    and return the checkpoint index.

    Parameters:
    -----------
        m_fp: FingerPrinter
            Initialized Fingerprinter model.
        checkpoint_dir: str
            Directory containing the checkpoints.
        checkpoint_index: int (default: 0)
            0 means the latest checkpoint. Epoch index starts from 1.

    Returns:
    --------
        checkpoint_index: int
            Index of the checkpoint loaded.

    """

    # Create a checkpoint and a checkpoint manager.
    checkpoint = tf.train.Checkpoint(model=m_fp)
    c_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)

    # Load the latest checkpoint if checkpoint_index is not specified.
    if checkpoint_index == 0:
        print("Argument 'checkpoint_index' was not specified.")
        print("Searching for the latest checkpoint...")
        latest_checkpoint = c_manager.latest_checkpoint
        if latest_checkpoint:
            checkpoint_index = int(latest_checkpoint.split(sep="ckpt-")[-1])
            status = checkpoint.restore(latest_checkpoint)
            status.expect_partial()
            print(f"=== Restored from {latest_checkpoint} ===")
        else:
            raise FileNotFoundError(f"Cannot find checkpoint in {checkpoint_dir}")
    # Load a particular checkpoint
    else:
        checkpoint_fpath = os.path.join(str(checkpoint_dir), f"ckpt-{checkpoint_index}")
        status = checkpoint.restore(checkpoint_fpath)  # Let TF to handle error cases.
        status.expect_partial()
        print(f"=== Restored from {checkpoint_fpath} ===")

    return checkpoint_index

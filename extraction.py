"""Extract fingerprints from a trained model. This script is different from the
evaluation-extraction.py script in that it only extracts fingerprints from given
audio files while the evaluation-extraction.py has extra functionalities designed
to evaluate model performance.

Also here, we process a single audio file at a time. This is not as fast as using
the evaluation-extraction.py script, but its more flexible in terms of audio inputs,
and it allows: sharding to multiple gpus and contiuing from a previous run.
The later is specifically useful when processing large databases, or when you want to
extend the database with new audio files without reprocessing the already processed ones.
"""

import os

# Create a new process group for the script
# to avoid killing the parent process when terminating child processes
# This is needed because in some machines, the tensorflow child processes
# are not killed when the script is terminated, which results in hanging.
os.setpgrp()

import gc
import traceback
import multiprocessing as mp
import signal
import time
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from nmfp.model.utils import get_fingerprinter, get_checkpoint_index_and_restore_model
from nmfp.gpu import set_gpu_memory_growth
from nmfp.dataloaders import InferenceDataset
from nmfp.utils import load_config, set_seed

# This rate was used when calculating the chunk boundaries, only used for future analysis
DATASET_SAMPLE_RATE = 8000


def infer(x: tf.Tensor, chunk_size: int) -> tf.Tensor:
    parts = tf.range(0, tf.shape(x)[0], chunk_size)
    outputs = []
    for start in parts:
        end = tf.minimum(start + chunk_size, tf.shape(x)[0])
        chunk = x[start:end]  # [b,1,F,T]
        outputs.append(compute_fp(chunk))  # single traced graph
    return tf.concat(outputs, axis=0)  # back to [N,d]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="""Directory containing multiple audio clips, a single audio clip, or 
        a line delimited text file where each line is a path to an audio clip. 
        We support .wav, .flac, .mp3, .aac, and .ogg audio formats. But the model was
        tested with .mp3 and .wav files only.""",
    )
    parser.add_argument(
        "config_path", type=Path, help="Path to the config file of the model."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory where the fingerprints will be stored. ",
    )
    parser.add_argument(
        "--hop-duration",
        type=float,
        default=0.5,
        help="Fingerprint generation rate in seconds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for model forward (#mel_spec_segments).",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=6,
        help="Number of workers for preparing the inputs.",
    )
    parser.add_argument(
        "--queue",
        "-q",
        type=int,
        default=24,
        help="Max queue size for preparing the inputs.",
    )
    parser.add_argument(
        "--block-growth",
        default=False,
        action="store_true",
        help="Allow GPU memory growth. I would allow growth. It speeds up considerably.",
    )
    parser.add_argument(
        "--num-partitions",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--partition",
        default=1,
        help="Partition number to process. 1 indexed!",
        type=int,
    )
    args = parser.parse_args()

    set_seed()

    tf.keras.backend.clear_session()

    assert 1 <= args.partition <= args.num_partitions, "Invalid partition parameters"

    if not args.block_growth:
        set_gpu_memory_growth()

    try:

        # Load the config file
        cfg = load_config(args.config_path)

        # Build the model
        m_fp = get_fingerprinter(cfg, trainable=False)

        # Wrap the model once with relaxed shapes:
        compute_fp = tf.function(
            m_fp,
            experimental_relax_shapes=True,  # ignore minor shape diffs
            reduce_retracing=True,  # bucket similar shapes
        )

        # Load the model weights
        checkpoint_dir = args.config_path.parent
        _ = get_checkpoint_index_and_restore_model(m_fp, checkpoint_dir)

        # Build the dataloader
        dataset = InferenceDataset(cfg)
        loader = dataset.get_loader(args.audio, args.hop_duration)

        # To preserve the relative paths of the audio files
        common_root = os.path.commonpath(loader.audio_paths)
        # Build the output paths and skip already processed files
        print("Skipping already processed files...")
        output_paths = {}
        for audio_path in loader.audio_paths:
            output_path = (
                args.output_dir
                / "embeddings"  # to group individual fingerprints, for the database mostly
                / audio_path.relative_to(common_root)  # preserve relative paths
            ).with_suffix(".npy")
            if not output_path.exists():
                output_paths[audio_path] = output_path
        loader.audio_paths = list(output_paths.keys())
        print(f"{len(output_paths):,} audio files remaining to be processed.")

        # Partition the audio files if requested
        if args.num_partitions > 1:
            output_paths = {
                k: output_paths[k]
                for k in list(output_paths.keys())[
                    args.partition - 1 :: args.num_partitions
                ]
            }
            loader.audio_paths = list(output_paths.keys())
            print(
                f"Partition {args.partition} will process {len(output_paths):,} files."
            )

        progbar = tf.keras.utils.Progbar(len(loader))

        try:

            """Parallelism in the dataloader to speed up processing------"""
            enq = tf.keras.utils.OrderedEnqueuer(
                loader, use_multiprocessing=True, shuffle=False
            )
            enq.start(workers=args.workers, max_queue_size=args.queue)
            gen = enq.get()
            start_time = time.monotonic()
            i, n = 0, 0
            try:
                while i < len(enq.sequence):
                    # Get the next batch of data
                    # X_mel is a 4D tensor of shape (batch_size, n_mels, n_frames, 1)
                    _, X_mel, X_path = next(gen)

                    if X_mel is None:
                        i += 1
                        progbar.update(i)
                        print(
                            f"\n\x1b[1;33m[WARNING] Skipping audio file {X_path} due to insufficient length.\x1b[0m"
                        )
                        continue

                    emb = infer(
                        X_mel,
                        chunk_size=args.batch_size,
                    )  # emb is a 2D tensor of shape (n_fingerprints, d)

                    # Write the fingerprints to disk
                    output_path = output_paths[X_path]
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(output_path, emb)

                    n += emb.shape[0]  # number of fingerprints
                    i += 1
                    progbar.update(i)
                progbar.update(i, finalize=True)
            except Exception as e:
                print(
                    "\n\x1b[1;31m[ERROR] An exception occurred during fingerprinting:\x1b[0m"
                )
                traceback.print_exc()  # full traceback
                raise e
        finally:
            enq.stop()  # guaranteed to run, even if next(gen) throws
        """ End of Parallelism----------------------------------------- """

        print(
            f"=== Processed {i:,} audio clips and stored {n:,} fingerprints in {args.output_dir} ==="
        )
        elapsed_time = time.strftime(
            "%H:%M:%S", time.gmtime(time.monotonic() - start_time)
        )
        print(f"Elapsed time for fingerprint extraction: {elapsed_time}")
        print("Done!")

    finally:
        # Keras + GC
        tf.keras.backend.clear_session()
        gc.collect()
        # Terminate any stray multiprocessing children
        for p in mp.active_children():
            p.terminate()
            p.join()
        print("All processes terminated.")
        os.killpg(os.getpgrp(), signal.SIGTERM)

"""As a primary step for pre-trained model evaluation, generate fingerprints.
It does some data processing so that after retrieval, query chunks can be
listened to in the context of the full track. By default it will use a hop
duration of 0.5 seconds, you can change it with the --hop-duration argument."""

import os

# Create a new process group for the script
# to avoid killing the parent process when terminating child processes
# This is needed because in some machines, the tensorflow child processes
# are not killed when the script is terminated, which results in hanging.
os.setpgrp()

import gc
import csv
import traceback
import time
import argparse
from pathlib import Path
from typing import Optional
import multiprocessing as mp
import signal

import numpy as np

import tensorflow as tf

from nmfp.model.utils import get_fingerprinter, get_checkpoint_index_and_restore_model
from nmfp.gpu import set_gpu_memory_growth
from nmfp.dataloaders import EvaluationDataset
from nmfp.utils import load_config, set_seed
from nmfp.lib_retrieval.parse_memmap import parse_memmap

# This rate was used when calculating the chunk boundaries, only used for future analysis
DATASET_SAMPLE_RATE = 8000


def main(
    config_path: Path,
    query_chunks: Path,
    db_tracks: Path,
    checkpoint_dir: Optional[Path],
    checkpoint_index: int,
    output_root_dir: Optional[Path],
    output_dir: Optional[Path],
    batch_size: int,
    flush_frequency: int,
    hop_duration: float,
    cpu_n_workers: int,
    cpu_max_que: int,
) -> None:
    """Generate fingerprints from a trained model checkpoint. Please
    check the argparse arguments for the details of the parameters."""

    # Load the config file
    cfg = load_config(config_path)

    # Get information from the config file
    checkpoint_name = cfg["MODEL"]["NAME"]
    log_root_dir = Path(cfg["MODEL"]["LOG_ROOT_DIR"])

    # Build the model
    m_fp = get_fingerprinter(cfg, trainable=False)

    # If checkpoint directory is not specified
    if checkpoint_dir is None:
        print("Checkpoint directory not specified. Using the config file.")
        # Try to read it from the config file
        checkpoint_dir = log_root_dir / "checkpoint" / checkpoint_name
        # If it does not exist, look next to the config file
        if not checkpoint_dir.is_dir():
            checkpoint_dir = Path(config_path).parent
            assert checkpoint_dir.is_dir(), f"Directory not found: {checkpoint_dir}"
    # Load checkpoint from checkpoint_dir using the epoch specified with checkpoint_index
    checkpoint_index = get_checkpoint_index_and_restore_model(
        m_fp, checkpoint_dir, checkpoint_index
    )

    """ Get the data loaders for fingerprinting """
    assert db_tracks or query_chunks, "At least one data source must be specified."
    ds = dict()
    dataset = EvaluationDataset(cfg)
    if query_chunks:
        print(f"Reading the query chunks from: {query_chunks}")
        loader = dataset.get_query_loader(query_chunks, batch_size, hop_duration)
        ds["queries"] = {
            "loader": loader,
            "boundary_paths": dataset.query_boundary_paths,
        }
    if db_tracks:
        print(f"Reading the database tracks from: {db_tracks}")
        loader = dataset.get_database_loader(db_tracks, batch_size, hop_duration)
        ds["database"] = {
            "loader": loader,
        }

    """ Determine the output directory """

    if output_dir is None:
        if output_root_dir is None:
            output_root_dir = log_root_dir / "emb"
        # If the output root directory name is not 'emb', add it
        if output_root_dir.name != "emb":
            output_root_dir = output_root_dir / "emb"
        # Add the generated fingerprints name to the output directory
        if "neural-music" in str(db_tracks) or "neural-music" in str(query_chunks):
            output_root_dir = output_root_dir / "nmfp"
        else:
            output_root_dir = output_root_dir / "other"
        output_dir = output_root_dir / checkpoint_name / str(checkpoint_index)

    # Create the output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    """ Generate fingerprints from each data source """
    for key, loader_dict in ds.items():
        start_time = time.monotonic()

        n_items = loader_dict["loader"].n_samples
        assert n_items > 0, f"Dataset '{key}' is empty."
        print(
            f"=== Generating \x1b[1;32m'{key}'\x1b[0m bsz={batch_size}, {n_items:,} items, d={m_fp.emb_sz} ==="
        )

        partition_dir = output_dir / key
        partition_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Saving fingerprints and track information to \x1b[1;32m{partition_dir}\x1b[0m"
        )

        # Track boundaries are used in
        # 1) parsing the block of concatenated query fingerprints
        # 2) working with the merged database
        track_paths, track_boundaries = loader_dict["loader"].get_track_information()
        csv_path = partition_dir / f"{key}.csv"

        """ Why use "memmap"?

        • First, we need to store a huge uncompressed embedding vectors until
        constructing a compressed DB with IVF-PQ (using FAISS). Handling a
        huge ndarray is not a memory-safe way: "memmap" consume 0 memory.

        • Second, Faiss-GPU does not support reconstruction of DB from
        compressed DB (index). In eval/eval_faiss.py, we need uncompressed
        vectors to calculate sequence-level matching score. The created
        "memmap" will be reused at that point.

        . Third, its much faster to write to a single memmap file
        than to write to multiple files.

        Reference:
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
        """

        # Create a memmap file
        arr_shape = (n_items, m_fp.emb_sz)
        arr_path = partition_dir / f"{key}.mm"
        arr = np.memmap(arr_path, dtype="float32", mode="w+", shape=arr_shape)

        progbar = tf.keras.utils.Progbar(len(loader_dict["loader"]))

        """ Parallelism to speed up processing------------------------- """
        enq = tf.keras.utils.OrderedEnqueuer(
            loader_dict["loader"], use_multiprocessing=True, shuffle=False
        )
        enq.start(workers=cpu_n_workers, max_queue_size=cpu_max_que)
        gen = enq.get()
        try:
            i = 0
            offset = 0
            while i < len(enq.sequence):
                progbar.update(i)
                _, Xa = next(gen)
                emb = m_fp(Xa)
                _bsz = emb.shape[0]
                arr[offset : offset + _bsz, :] = emb.numpy()
                i += 1
                offset += _bsz
                if i % flush_frequency == 0 or i == len(enq.sequence):
                    arr.flush()
            progbar.update(i, finalize=True)
        finally:
            enq.stop()  # guaranteed to run, even if next(gen) throws
        """ End of Parallelism----------------------------------------- """
        print(
            f"=== Succesfully stored {len(arr):,} {key} fingerprints to {partition_dir} ==="
        )

        # Close the memmap file
        del arr

        elapsed_time = time.strftime(
            "%H:%M:%S", time.gmtime(time.monotonic() - start_time)
        )
        print(f"Elapsed time for fingerprint generation: {elapsed_time}")

        # Save the csv file, do some post-processing for the queries
        if key == "queries":
            # Parse the fingerprints of the memmap to individual tracks
            parse_memmap(
                mm_path=arr_path,
                mm_shape=arr_shape,
                track_paths=track_paths,
                track_boundaries=track_boundaries,
                output_dir=partition_dir,
                delete_original=True,
            )

            # Write the boundaries of the sampled query chunks inside the full tracks
            # This is used for segment-level evaluation
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["audio_path", "start_segment", "end_segment"])
                if loader_dict["boundary_paths"]:
                    # This is a very controlled situation, which occurs for our dataset
                    for boundary_path in loader_dict["boundary_paths"]:
                        f_path = boundary_path.replace(".npy", ".wav")  # NOTE dirty fix
                        start_idx, end_idx = np.load(boundary_path)
                        start_time = start_idx / DATASET_SAMPLE_RATE
                        end_time = end_idx / DATASET_SAMPLE_RATE
                        t = hop_duration * np.arange(0, (end_time // hop_duration) + 5)
                        start_time = t[np.abs(t - start_time).argmin()]
                        end_time = t[np.abs(t - end_time).argmin()]
                        start_idx = int(start_time / hop_duration)
                        end_idx = int(end_time / hop_duration)
                        writer.writerow([f_path, start_idx, end_idx])
                else:
                    # In the industrial data we do not have the chunk boundaries
                    for f_path in ds["queries"]["query_chunk_paths"]:
                        writer.writerow([f_path])
        else:
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                # emb dim is redundant but simplifies the loading process
                writer.writerow(
                    ["audio_path", "start_segment", "end_segment", "emb_dim"]
                )
                for f_path, (start, end) in zip(track_paths, track_boundaries):
                    writer.writerow([f_path, start, end, m_fp.emb_sz])
        print(f"CSV file created at {str(csv_path)}")

    print()
    print("\x1b[1;32m=== Fingerprinting completed ===\x1b[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_path", type=Path, help="Path to the config file of the model."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="""Directory containing the query audio or a line delimited 
        text file containing paths.""",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="""Directory containing the database tracks or a line delimited 
        text file containing paths.""",
    )
    parser.add_argument(
        "--hop-duration",
        type=float,
        default=0.5,
        help="Fingerprint generation rate in seconds.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="""Directory containing the checkpoints. If not provided, it 
        will first check the config file for a path. If not found, it will 
        look next to the config file.""",
    )
    parser.add_argument(
        "--checkpoint-index",
        type=int,
        default=0,
        help="Checkpoint index. 0 means the latest checkpoint.",
    )
    parser.add_argument(
        "--output-root-dir",
        type=Path,
        default=None,
        help="""Root directory where the generated fingerprints will 
        be stored. If not specified, it will be saved in the log directory 
        of the model in the config. Following the structure: 
        log_root_dir/emb/model_name/checkpoint_index/""",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="""Output directory where the fingerprints will be stored. 
        If not specified, it will be saved in the output_root_dir. 
        If provided output_root_dir will be ignored.""",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="""Batch size for inference (#mel_spec_segments). 
        A batch can contain segments from different tracks.""",
    )
    parser.add_argument(
        "--flush-frequency",
        type=int,
        default=500,
        help="""To speed up the process, the fingerprints are written to the 
        disk in batches. The flush frequency determines how often the memmap 
        is flushed to the disk. The higher the frequency, the faster the 
        process but the more memory is used.""",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=6,
        help="Number of workers for audio loading.",
    )
    parser.add_argument(
        "--queue", "-q", type=int, default=24, help="Max queue size for audio loading."
    )
    parser.add_argument(
        "--block-growth",
        default=False,
        action="store_true",
        help="Allow GPU memory growth. I would allow growth. It speeds up considerably.",
    )
    args = parser.parse_args()

    set_seed()

    tf.keras.backend.clear_session()

    if not args.block_growth:
        set_gpu_memory_growth()

    # Generate fingerprints
    # This ugly block is to ensure that the script
    # does not hang on exit due to multiprocessing issues
    # and it prints the traceback of the exception
    try:
        try:
            main(
                args.config_path,
                query_chunks=args.queries,
                db_tracks=args.database,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_index=args.checkpoint_index,
                output_root_dir=args.output_root_dir,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                flush_frequency=args.flush_frequency,
                hop_duration=args.hop_duration,
                cpu_n_workers=args.workers,
                cpu_max_que=args.queue,
            )
        except Exception as e:
            print(
                "\n\x1b[1;31m[ERROR] An exception occurred during fingerprinting:\x1b[0m"
            )
            traceback.print_exc()  # full traceback
            raise e
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

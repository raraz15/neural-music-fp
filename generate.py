"""Generate fingerprints from a trained model."""

import os
import time
import argparse

import numpy as np

import tensorflow as tf

from nmfp.model.utils import get_fingerprinter, get_checkpoint_index_and_restore_model
from nmfp.model.nnfp import FingerPrinter
from nmfp.dataset_eval import EvaluationDataset
from nmfp.gpu import choose_first_gpu
from nmfp.utils import load_config, set_seed


def get_data_source(
    cfg: dict,
    db_tracks: str,
    query_chunks: str,
    batch_sz: int,
    hop_duration: float,
) -> dict:
    """Get the data loaders for fingerprinting. The data source must be one of the
    the query chunks or the database tracks.

    Parameters
    ----------
        cfg : dict
            A dictionary containing the model configurations
        db_tracks : str, optional
            Directory containing the database tracks or a line delimited text
            file containing paths.
        query_chunks : str, optional
            Directory containing the query chunks or a line delimited text
            file containing paths.
        batch_sz : int, optional
            Batch size for inference, number of audio seegment inputs.
        hop_duration : float, optional
            Fingerprint generation rate in seconds.

    Returns
    -------
        ds : dict
            Dictionary containing the dataloaders.
    """

    if db_tracks:
        print(f"Database tracks will be read from: {db_tracks}")
    if query_chunks:
        print(f"Query chunks will be read from: {query_chunks}")

    assert db_tracks or query_chunks, "At least one data source must be specified."

    ds = dict()
    dataset = EvaluationDataset(cfg)

    if query_chunks:
        loader = dataset.get_query_loader(query_chunks, batch_sz, hop_duration)
        ds["query_chunks"] = {
            "output_dir": "query",
            "loader": loader,
            "boundary_paths": dataset.query_boundary_paths,
        }

    if db_tracks:
        loader = dataset.get_database_loader(db_tracks, batch_sz, hop_duration)
        ds["db_tracks"] = {
            "output_dir": "database",
            "loader": loader,
        }

    return ds


@tf.function
def test_step(X, m_fp: FingerPrinter):
    """Test step used for generating fingerprints. Not sure if @tf.function
    makes a real difference in terms of performance.

    Parameters
    ----------
        X: Input tensor. X is not a tuple of (Xa, Xp) here as in train_step().
        m_fp: FingerPrinter
            Model object.

    Returns
    -------
        fp: Fingerprint tensor of dim (BSZ, Dim)
    """

    # Set the model to inference mode
    m_fp.trainable = False

    # Forward pass to create the fingerprint
    fp = m_fp(X)

    return fp


def parse_memmap(
    arr_path: str,
    arr_shape_path: str,
    partition_dir: str,
    key: str,
    track_paths: list,
    track_boundaries: list,
    arr_shape: tuple,
    delete_original: bool = True,
):
    """Parse the fingerprints of the mixture memmap to individual tracks and write them
    to disk as individual files. The fingerprints will be saved as individual memmap files
    in case of database tracks and as individual npy files in case of query chunks. The
    original memmap file will be deleted if specified."""

    # Time the parsing
    start_time = time.monotonic()

    # Load the memmap file with read-only mode
    arr = np.memmap(arr_path, dtype="float32", mode="r", shape=arr_shape)

    # Parse the fingerprints
    print(f"=== Parsing the mixture memmap to individual tracks ===")
    for track_path, boundary in zip(track_paths, track_boundaries):
        # Get the track id
        track_id = os.path.splitext(os.path.basename(track_path))[0]
        # Group fingerprints by id[:3]
        track_dir = os.path.join(partition_dir, track_id[:3])
        os.makedirs(track_dir, exist_ok=True)
        # Load the fingerprints of the track from the memmap
        fp = arr[boundary[0] : boundary[1], :]
        if key == "db_tracks":  # Save db_tracks as individual memmap files
            # Create a new memmap file and copy the data
            new_memmap = np.memmap(
                os.path.join(track_dir, f"{track_id}.mm"),
                dtype="float32",
                mode="w+",
                shape=fp.shape,
            )
            new_memmap[:] = fp
            new_memmap.flush()
            del new_memmap
            # Write the shape of the memmap
            np.save(os.path.join(track_dir, f"{track_id}-shape.npy"), fp.shape)
        elif key == "query_chunks":  # Save query chunks as individual npy files
            # Convert to numpy array
            fp = np.array(fp)
            # Save query fingerprints
            np.save(os.path.join(track_dir, f"{track_id}.npy"), fp)
        else:
            raise ValueError(f"Unknown key: {key}")

    # Close memmap
    del arr

    # Print the elapsed time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.monotonic() - start_time))
    print(f"Elapsed time for parsing the fingerprints: {elapsed_time}")

    # Remove the original memmap file if specified
    if delete_original:
        print("Deleting the original memmap file...")
        os.remove(arr_path)
        os.remove(arr_shape_path)


def main(
    config_path: str,
    query_chunks: str,
    db_tracks: str,
    checkpoint_dir: str,
    checkpoint_index: int,
    output_root_dir: str,
    output_dir: str,
    batch_sz: int,
    hop_duration: float,
    mixed_precision: bool,
    cpu_n_workers: int,
    cpu_max_que: int,
) -> None:
    """Generate fingerprints from a trained model checkpoint. Please
    check the argparse arguments for the details of the parameters."""

    # Load the config file
    cfg = load_config(config_path)

    # Get information from the config file
    checkpoint_name = cfg["MODEL"]["NAME"]
    log_root_dir = cfg["MODEL"]["LOG_ROOT_DIR"]  # Can be overwritten by the arguments
    dim = cfg["MODEL"]["ARCHITECTURE"]["EMB_SZ"]
    sample_rate = cfg["MODEL"]["AUDIO"]["FS"]

    # Set mixed precision before building the model
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled.")
        cfg["TRAIN"]["MIXED_PRECISION"] = True
    else:
        print("Using full precision.")
        cfg["TRAIN"]["MIXED_PRECISION"] = False

    # Build the model
    m_fp = get_fingerprinter(cfg, trainable=False)

    # If checkpoint directory is not specified
    if checkpoint_dir is None:
        print("Checkpoint directory not specified. Using the config file.")
        # Try to read it from the config file
        checkpoint_dir = os.path.join(log_root_dir, "checkpoint", checkpoint_name)
        # If it does not exist, look next to the config file
        if not os.path.isdir(checkpoint_dir):
            print(f"Looking next to the config for the checkpoint.")
            checkpoint_dir = os.path.dirname(config_path)
            assert os.path.isdir(
                checkpoint_dir
            ), f"Checkpoint directory not found: {checkpoint_dir}"

    # Load checkpoint from checkpoint_dir using the epoch specified with checkpoint_index
    checkpoint_index = get_checkpoint_index_and_restore_model(
        m_fp, checkpoint_dir, checkpoint_index
    )

    """ Determine the output directory """

    if output_dir is None:
        if output_root_dir is None:
            output_root_dir = os.path.join(log_root_dir, "fp")
        # If the output root directory name is not fp, add it
        if os.path.basename(os.path.normpath(output_root_dir)) != "fp":
            output_root_dir = os.path.join(output_root_dir, "fp")
        # Output dir is output_root_dir/checkpoint_name/checkpoint_index/
        output_dir = os.path.join(
            output_root_dir, checkpoint_name, str(checkpoint_index)
        )

    # Write the precision type of the inference
    if mixed_precision:
        output_dir = os.path.join(output_dir, "mixed_precision")

    # Check if the output directory exists
    if os.path.isdir(output_dir):
        print(f"Output directory already exists: {output_dir}")
        print("Please specify a different output directory.")
        raise FileExistsError

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    """ Get the data loaders for fingerprinting """

    ds = get_data_source(
        cfg,
        db_tracks=db_tracks,
        query_chunks=query_chunks,
        batch_sz=batch_sz,
        hop_duration=hop_duration,
    )

    """ Generate fingerprints from each data source """

    for key, loader_dict in ds.items():
        # Time the current data source
        start_time = time.monotonic()

        # Check if the dataset is empty
        n_items = loader_dict["loader"].n_samples
        assert n_items > 0, f"Dataset '{key}' is empty."
        print(
            f"=== Generating \x1b[1;32m'{key}'\x1b[0m bsz={batch_sz}, {n_items:,} items, d={dim} ==="
        )

        # Make the output directory for the partition
        partition_dir = os.path.join(output_dir, loader_dict["output_dir"])
        os.makedirs(partition_dir, exist_ok=True)
        print(
            f"Saving fingerprints and track information to \x1b[1;32m{partition_dir}\x1b[0m"
        )

        # Save track information
        track_paths, track_boundaries = loader_dict["loader"].get_track_information()
        np.save(os.path.join(partition_dir, "track_boundaries.npy"), track_boundaries)
        with open(os.path.join(partition_dir, "track_paths.txt"), "w") as f:
            f.write("\n".join(track_paths) + "\n")

        # Create memmap, fill it with fingerprints, and save it to disk
        """ Why use "memmap"?

        • First, we need to store a huge uncompressed embedding vectors until
          constructing a compressed DB with IVF-PQ (using FAISS). Handling a
          huge ndarray is not a memory-safe way: "memmap" consume 0 memory.

        • Second, Faiss-GPU does not support reconstruction of DB from
          compressed DB (index). In eval/eval_faiss.py, we need uncompressed
          vectors to calculate sequence-level matching score. The created
          "memmap" will be reused at that point.

        Reference:
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
        """

        # Create a memmap file
        arr_shape = (n_items, dim)
        arr_path = os.path.join(partition_dir, "fingerprints.mm")
        arr_shape_path = os.path.join(partition_dir, "shape.npy")
        arr = np.memmap(arr_path, dtype="float32", mode="w+", shape=arr_shape)
        # Save the shape of the memmap
        np.save(arr_shape_path, arr_shape)

        # Fingerprinting loop
        progbar = tf.keras.utils.Progbar(len(loader_dict["loader"]))

        """ Parallelism to speed up processing------------------------- """
        enq = tf.keras.utils.OrderedEnqueuer(
            loader_dict["loader"], use_multiprocessing=True, shuffle=False
        )
        enq.start(workers=cpu_n_workers, max_queue_size=cpu_max_que)

        i = 0
        while i < len(enq.sequence):
            progbar.update(i)
            _, Xa = next(enq.get())
            emb = test_step(Xa, m_fp)
            _bsz = emb.shape[0]
            arr[i * _bsz : (i + 1) * _bsz, :] = emb.numpy()
            i += 1
            # Flush the memmap every iteration
            arr.flush()
        progbar.update(i, finalize=True)
        enq.stop()
        """ End of Parallelism----------------------------------------- """

        # Write all the fingerprints to disk as a single memmap file
        arr.flush()

        # Print summary
        print(
            f"=== Succesfully stored {len(arr):,} {key} fingerprints to {partition_dir} ==="
        )

        # Close memmap
        del arr

        # Print the elapsed time
        elapsed_time = time.strftime(
            "%H:%M:%S", time.gmtime(time.monotonic() - start_time)
        )
        print(f"Elapsed time for fingerprint generation: {elapsed_time}")

        if key == "query_chunks":
            # Parse the fingerprints of the memmap to individual tracks
            parse_memmap(
                arr_path,
                arr_shape_path,
                partition_dir,
                key,
                track_paths,
                track_boundaries,
                arr_shape,
            )

            # Copy the boundaries of the sampled query chunks in the full tracks
            for original_boundary in loader_dict["boundary_paths"]:
                # Get the track id
                track_id = os.path.splitext(os.path.basename(original_boundary))[0]
                # Group fingerprints by id[:3]
                track_dir = os.path.join(partition_dir, track_id[:3])

                # Load the boundary file
                start_time, end_time = np.load(original_boundary) / sample_rate

                # Create a long time array
                t = hop_duration * np.arange(0, (end_time // hop_duration) + 5)

                # Find the gt index
                start_time = t[np.abs(t - start_time).argmin()]
                gt_idx = int(start_time / hop_duration)

                # Save the chunk's boundary inside the track
                np.save(
                    os.path.join(track_dir, f"{track_id}-chunk_boundary_idx.npy"),
                    np.array(gt_idx),
                )

    print()
    print("\x1b[1;32m=== Fingerprinting completed ===\x1b[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the config file of the model."
    )
    parser.add_argument(
        "--query_chunks",
        type=str,
        default=None,
        help="Directory containing the query chunks or a line delimited text "
        "file containing paths.",
    )
    parser.add_argument(
        "--db_tracks",
        type=str,
        default=None,
        help="Directory containing the database tracks or a line delimited text "
        "file containing paths.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing the checkpoints. If not provided, it will "
        "first check the config file for a path. If not found, it will look next "
        "to the config file.",
    )
    parser.add_argument(
        "--checkpoint_index",
        type=int,
        default=0,
        help="Checkpoint index. 0 means the latest checkpoint.",
    )
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default=None,
        help="Root directory where the generated fingerprints will be stored."
        "If not specified, it will be saved in the log directory of the model in "
        "the config. Following the structure: log_root_dir/fp/model_name/checkpoint_index/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory where the fingerprints will be stored. "
        "If not specified, it will be saved in the output_root_dir. "
        "If provided output_root_dir will be ignored.",
    )
    parser.add_argument(
        "--batch_sz", type=int, default=256, help="Batch size for inference."
    )
    parser.add_argument(
        "--hop_duration",
        type=float,
        default=0.5,
        help="Fingerprint generation rate in seconds.",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision during inference. "
        "The fingerprints will be saved in FP32 in both cases.",
    )
    parser.add_argument(
        "--cpu_n_workers",
        type=int,
        default=10,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--cpu_max_que", type=int, default=10, help="Max queue size for data loading."
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed()

    # Ensure that the first GPU is selected
    choose_first_gpu()

    # Generate fingerprints
    main(
        args.config_path,
        query_chunks=args.query_chunks,
        db_tracks=args.db_tracks,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_index=args.checkpoint_index,
        output_root_dir=args.output_root_dir,
        output_dir=args.output_dir,
        batch_sz=args.batch_sz,
        hop_duration=args.hop_duration,
        mixed_precision=args.mixed_precision,
        cpu_n_workers=args.cpu_n_workers,
        cpu_max_que=args.cpu_max_que,
    )

"""Segment/sequence-wise music search and evaluation: implementation based on FAISS."""

import os
import time
import glob
import argparse

from typing import Tuple, List

import numpy as np
import pandas as pd

from nmfp.get_index_faiss import get_index
from nmfp.utils import set_seed, get_track_id_filled


def load_db_fingerprints(database_dir: str) -> dict:
    """Load the unified database fingerprints and metadata.

    Parameters:
    -----------
    database_dir: str
        The directory containing the database fingerprints and related metadata.

    Returns:
    --------
    db_fp: dict
        A dictionary containing the database fingerprints and metadata.
            db_fp["track_boundaries"]: np.ndarray
                The boundaries of the tracks in the database fingerprints.
            db_fp["track_paths"]: list
                The paths to the tracks in the database.
            db_fp["data"]: np.memmap
                The merged database fingerprints.
            db_fp["data_shape"]: tuple
                The shape of the merged database fingerprints.
            db_fp["path"]: str
                The path to the merged database fingerprints.
    """

    print("Loading fingerprints from the Database...")

    # Dict to store the database fingerprints and metadata
    db_fp = {}

    # Load all the track boundaries and paths
    db_fp["track_boundaries"] = np.load(
        os.path.join(database_dir, "track_boundaries.npy")
    )
    with open(os.path.join(database_dir, "track_paths.txt"), "r") as in_f:
        db_fp["track_paths"] = [p.strip() for p in in_f.read().splitlines()]
    assert len(db_fp["track_boundaries"]) == len(
        db_fp["track_paths"]
    ), f"Track boundaries and paths mismatch: {len(db_fp['track_boundaries'])} != {len(db_fp['track_paths'])}"
    print(
        f"{len(db_fp['track_paths']):>10,} database tracks in \033[32m{database_dir}\033[0m"
    )

    print("Loading the single memmap of the Database...")

    # Load the unified database memmap
    db_path = os.path.join(database_dir, "fingerprints.mm")
    db_shape = tuple(np.load(os.path.join(database_dir, "shape.npy")))
    db = np.memmap(
        db_path,
        dtype="float32",
        mode="r",
        shape=db_shape,
    )

    db_fp["path"] = db_path
    db_fp["data_shape"] = db_shape
    db_fp["data"] = db

    return db_fp


def load_query_fingerprints(query_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """Load the query fingerprints and track paths. The query fingerprints are
    loaded as individual numpy files.

    Parameters:
    -----------
    query_dir: str
        The directory containing the query fingerprints. It should contain
        "track_paths.txt" file.

    Returns:
    --------
    query_fp_paths: list
        The paths to the query fingerprints.
    q_chunk_bound_paths: list
        The paths to the query chunks' boundary indices in the full tracks.
    query_track_paths: list
        The paths to the query tracks.
    """

    print("Loading the query fingerprints...")

    # Get the fingerprint paths
    query_fp_paths = sorted(
        glob.glob(os.path.join(query_dir, "**", "*.npy"), recursive=True)
    )
    # Filter out "track_boundaries.npy" and "chunk_boundary_idx.npy"
    query_fp_paths = [
        p
        for p in query_fp_paths
        if "track_boundaries.npy" != os.path.basename(p)
        and "-chunk_boundary_idx.npy" not in p
    ]
    print(
        f"{len(query_fp_paths):>10,} query chunks' fingerprints found in \033[32m{query_dir}\033[0m"
    )
    assert len(query_fp_paths) > 0, f"No query fingerprints found in {query_dir}"

    # Read the gt segment indices of the query chunks
    q_chunk_bound_paths = [
        p.replace(".npy", "-chunk_boundary_idx.npy") for p in query_fp_paths
    ]
    for p in q_chunk_bound_paths:
        assert os.path.exists(p), f"Query segment index file not found: {p}"

    # Read the query track paths
    with open(os.path.join(query_dir, "track_paths.txt"), "r") as in_f:
        query_track_paths = sorted([p.strip() for p in in_f.read().splitlines()])
    # Make sure that the query track IDs and the fingerprint IDs match
    for q_fp_path, q_track_path in zip(query_fp_paths, query_track_paths):
        q_fp_id = get_track_id_filled(q_fp_path)
        q_track_id = get_track_id_filled(q_track_path)
        assert (
            q_fp_id == q_track_id
        ), f"Query fingerprint and track ID mismatch: {q_fp_id} != {q_track_id}"

    return query_fp_paths, q_chunk_bound_paths, query_track_paths


def main(
    query_dir: str,
    database_dir: str,
    output_root_dir: str,
    output_dir: str,
    index_type: str,
    max_train: int,
    top_k: int,
    n_probe: int,
    segment_duration: float,
    hop_duration: float,
    test_seq_len: str,
    delta_n: int,
    display_interval: int,
    no_gpu: bool,
) -> None:
    """For a detailed explanation of the arguments, please refer to the argparse
    section at the end of this script."""

    """ Check the arguments """
    assert (
        max_train is None or max_train > 0
    ), f"max_train must be None or > 0: {max_train}"
    assert top_k > 0, f"top_k must be > 0: {top_k}"
    assert n_probe > 0, f"n_probe must be > 0: {n_probe}"
    assert segment_duration > 0, f"segment_duration must be > 0: {segment_duration}"
    assert hop_duration > 0, f"hop_duration must be > 0: {hop_duration}"
    assert delta_n > 0, f"delta_n must be > 0: {delta_n}"
    assert display_interval > 0, f"display_interval must be > 0: {display_interval}"

    """ Determine the output directory"""

    fingerprints_dir = os.path.dirname(os.path.normpath(query_dir))

    if output_dir is None:

        # If output_root_dir is not provided, set it to 3 levels above the fingerprints_dir
        if output_root_dir is None:
            output_root_dir = os.path.abspath(
                os.path.join(
                    fingerprints_dir,
                    os.path.pardir,
                    os.path.pardir,
                    os.path.pardir,
                    "eval",
                )
            )

        # If the output_root_dir is not named "eval", add it
        if os.path.basename(os.path.normpath(output_root_dir)) != "eval":
            output_root_dir = os.path.join(output_root_dir, "eval")

        # Determine the output directory
        epoch = os.path.basename(os.path.normpath(fingerprints_dir))
        model_name = os.path.basename(
            os.path.dirname(os.path.normpath(fingerprints_dir))
        )
        output_dir = os.path.join(output_root_dir, model_name, epoch)

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: \033[32m{output_dir}\033[0m")

    """ Deal with the test_seq_len """

    # '1,3,5' --> [1, 3, 5] and sort
    test_seq_len = np.asarray(sorted(list(map(int, test_seq_len.split(",")))))
    assert np.all(test_seq_len > 0), f"test_seq_len must be > 0: {test_seq_len}"

    """ Get query fingerprint and track paths """

    query_fp_paths, q_chunk_bound_paths, query_track_paths = load_query_fingerprints(
        query_dir
    )

    """ Create the fingerprint database and index """

    db_fp = load_db_fingerprints(database_dir)
    print(f"Database contains {len(db_fp['track_paths']):>10,} full length tracks.")

    # Create and train FAISS index. Use all the db for training the index
    # Using the relevant and irrelevant tracks together for training or limiting the
    # number of training tracks affects the performance below 0.1%.
    index = get_index(
        train_data=db_fp["data"],
        index_type=index_type,
        max_nitem_train=max_train,
        n_probe=n_probe,
        use_gpu=(not no_gpu),
    )

    # Since the index is created and trained, we don't need the db fingerprints anymore
    del db_fp["data"]

    """ ----------------------------------------------------------------------

    • Calculation of sequence-level matching score requires reconstruction of
      vectors from FAISS index.
    • Unfortunately, current faiss.index.reconstruct_n(id_start, id_stop)
      supports only CPU index.
    • We prepare a fake_recon_index through the on-disk method.

    ---------------------------------------------------------------------- """

    # Prepare fake_recon_index
    print(f"Preparing fake_recon_index...")
    fake_recon_index = np.memmap(
        db_fp["path"],
        dtype="float32",
        mode="r",
        shape=db_fp["data_shape"],
    )
    print(f"Created fake_recon_index, total {db_fp['data_shape'][0]:>10,} items.")

    """ Segment/sequence-level search & evaluation """

    # Define metrics, tables, and variables
    top1_song, top1_exact, top1_near, top1_far, analysis = [], [], [], [], []
    start_time, total_search_time = time.monotonic(), 0

    # Start printing the table, convert sequence lengths to time
    _str = (
        "||"
        + "|".join(
            "{:^9}".format((arg - 1) * hop_duration + segment_duration)
            for arg in test_seq_len
        )
        + "||"
    )  # |  xy.zt  |
    pretty_print_len = len(_str)
    print(
        f'\n{" Top-1 Track Hit Rate (%) of Sequence-Level Search ":=^{pretty_print_len}}\n'
    )
    print(f'{" Query Sequence Duration (s) ":-^{pretty_print_len}}')
    print(_str)
    print("-" * pretty_print_len)

    # For each query audio file's fingerprints
    for i, (q_fp_path, q_chunk_bound_path, q_track_path) in enumerate(
        zip(query_fp_paths, q_chunk_bound_paths, query_track_paths)
    ):
        # FMA id of the query fingerprint
        q_track_ID = get_track_id_filled(q_fp_path)
        # FMA id of the ground truth track
        gt_track_ID = q_track_ID  # NOTE: We ensure this for FMA only!
        # Get the path to the ground truth track
        gt_track_path = [
            track_path
            for track_path in db_fp["track_paths"]
            if get_track_id_filled(track_path) == q_track_ID
        ][0]
        # The start idx of the query chunk in the full track
        q_chunk_bound = np.load(q_chunk_bound_path)

        # Load the fingerprints of the query audio to memory
        q_fp = np.load(q_fp_path)

        # Make sure each query sequence is shorter than the query chunk and is equally spaced
        assert q_fp.shape[0] >= test_seq_len[-1], (
            f"Query sequence length is shorter than the query chunk: "
            f"{q_fp.shape[0]} < {test_seq_len[-1]}"
        )
        assert (
            delta_n < q_fp.shape[0]
        ), f"delta_n must be less than the query chunk: {delta_n} < {q_fp.shape[0]}"
        sequence_start_indices = np.arange(0, len(q_fp) + 1 - test_seq_len[-1], delta_n)

        # Get a sequence inside the query from each of the sequence_start_indices
        for j, seq_start_idx in enumerate(sequence_start_indices):
            # Initialize the lists for each sequence
            top1_song.append(np.zeros(len(test_seq_len)).astype(int))
            top1_exact.append(np.zeros(len(test_seq_len)).astype(int))
            top1_near.append(np.zeros(len(test_seq_len)).astype(int))
            top1_far.append(np.zeros(len(test_seq_len)).astype(int))

            # The current sequence starts from this idx in the full track
            q_start_segment = q_chunk_bound + seq_start_idx

            # For each sequence length
            for k, seq_len in enumerate(test_seq_len):
                # Get the query sequence. if seq_len=1, query is a single fingerprint.
                # Otherwise its a sequence of fingerprints.
                q = q_fp[seq_start_idx : seq_start_idx + seq_len, :]

                # For each segment in the sequence q, get the top k most similar
                # segments from the database, independently.
                _, I = index.search(q, top_k)  # _: distance, I: result IDs matrix

                """Is uses the continuity of music. If the ith query segment matches a
                jth segment in the database, the (i+1)th query segment should match the
                (j+1)th segment in the database. Using this assumption, create candidate 
                sequences in the database. For each candidate segment, get a same length 
                sequence starting at from the past and including the candidate segment. 
                Calculate the similarity of the query sequence and the candidate sequence 
                then average the similarity scores."""
                for offset in range(len(I)):
                    I[offset, :] -= offset

                # Find the unique candidates that are not -1
                candidates = np.unique(I[np.where(I >= 0)])

                # Calculate the average similarity score for each candidate sequence
                _scores = np.zeros(len(candidates))
                for ci, cid in enumerate(candidates):
                    _scores[ci] = np.mean(
                        np.diag(
                            # np.dot(q, index.reconstruct_n(cid, (cid + l)).T)
                            np.dot(q, fake_recon_index[cid : cid + seq_len, :].T)
                        )
                    )

                """ Evaluate the sequence-level search in terms of song hit """

                # Position of the top fingerprint inside the DB
                # Each candidate referes to the start segment of when
                # the query sequence was initiated
                _idx = np.argsort(-_scores)[0]
                score = _scores[_idx]

                # Starting idx of the top fingerprint inside the DB
                p_fp_idx = candidates[_idx]

                # Position of the predicted track inside the DB
                p_track_idx = np.where(
                    (p_fp_idx >= db_fp["track_boundaries"][:, 0])
                    & (p_fp_idx < db_fp["track_boundaries"][:, 1])
                )[0][0]
                # Position of the top predicted segment inside the track (for listening)
                p_start_segment = p_fp_idx - db_fp["track_boundaries"][p_track_idx, 0]

                # Path to the top prediction
                p_track_path = db_fp["track_paths"][p_track_idx]
                # Track ID of the top prediction
                p_track_ID = os.path.splitext(os.path.basename(p_track_path))[0]

                # Check if the top predicted track is correct
                top1_song[-1][k] = int(gt_track_ID == p_track_ID)

                # Check if the top predicted segment is aligned in time
                if top1_song[-1][k] == 1:
                    top1_exact[-1][k] = int(p_start_segment == q_start_segment)
                    top1_near[-1][k] = int(abs(p_start_segment - q_start_segment) <= 1)
                    top1_far[-1][k] = 1  # If it is not exact or near, it is far

                # Create a table to analyze the results later.
                analysis.append(
                    {
                        "query_track_path": q_track_path,
                        "query_start_segment": q_start_segment,  # inside the full track
                        "query_chunk_bound": q_chunk_bound,  # inside the full track
                        "seq_start_idx": seq_start_idx,  # inside the chunk
                        "seq_len": seq_len,
                        "gt_track_path": gt_track_path,
                        "pred_track_path": p_track_path,
                        "pred_start_segment": p_start_segment,
                        "score": score,
                    }
                )

        # Print summary
        if ((i + 1) % display_interval) == 0 or (
            i == len(query_fp_paths) - 1
            and j == len(sequence_start_indices) - 1
            and k == len(test_seq_len) - 1
        ):
            # Calculate the top1 song rate
            top1_song_rate = np.stack(top1_song, axis=0)
            top1_song_rate = np.round(100 * np.mean(top1_song_rate, axis=0), 2)

            # Calculate the top1 segment hit rates
            top1_exact_rate = np.stack(top1_exact, axis=0)
            top1_near_rate = np.stack(top1_near, axis=0)
            top1_far_rate = np.stack(top1_far, axis=0)
            top1_exact_rate = np.round(100 * np.mean(top1_exact_rate, axis=0), 2)
            top1_near_rate = np.round(100 * np.mean(top1_near_rate, axis=0), 2)
            top1_far_rate = np.round(100 * np.mean(top1_far_rate, axis=0), 2)

            # Print the table
            interval_time = time.monotonic() - start_time
            avg_search_time = interval_time / display_interval  # s
            print(
                "||" + "|".join("{:^9.2f}".format(arg) for arg in top1_song_rate),
                end="",
            )
            print(f"|| {avg_search_time :>4.2f} s/track ", end="")
            print(f"[{i+1:>{len(str(len(query_fp_paths)))}}/{len(query_fp_paths)}] ||")
            total_search_time += interval_time
            start_time = time.monotonic()  # reset interval stopwatch

    print("-" * (pretty_print_len + 15) + "\n")

    # Delete fake_recon_index
    del fake_recon_index

    print(
        f"Finished all search in {time.strftime('%H:%M:%S', time.gmtime(total_search_time))}",
    )

    """ Save results """

    # Save the matching results
    df = pd.DataFrame(analysis)
    df.to_csv(os.path.join(output_dir, "analysis.csv"), index=False)

    # Save the scores and test_ids
    with open(os.path.join(output_dir, "track_hit_rate.txt"), "w") as out_f:
        out_f.write(" & ".join(map(str, test_seq_len)) + "\n")
        out_f.write(" & ".join(map(str, top1_song_rate)) + "\n")
    with open(os.path.join(output_dir, "segment_hit_rate.txt"), "w") as out_f:
        out_f.write("          " + "  & ".join(map(str, test_seq_len)) + "\n")
        out_f.write("Exact:\t" + " & ".join(map(str, top1_exact_rate)) + "\n")
        out_f.write(" Near:\t" + " & ".join(map(str, top1_near_rate)) + "\n")
        out_f.write("  Far:\t" + " & ".join(map(str, top1_far_rate)) + "\n")
    print(f"Saved the scores to {output_dir}")

    # Print the contents of segment_hit_rate.txt
    print("Segment-level Top1-Hit Rates:")
    with open(os.path.join(output_dir, "segment_hit_rate.txt"), "r") as in_f:
        print(in_f.read())

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "query_dir",
        type=str,
        help="Path to the query directory.",
    )
    parser.add_argument(
        "database_dir",
        type=str,
        help="Path to the database directory.",
    )
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default=None,
        help="Output root directory. Inside the root directory, the results will be saved "
        "in output_root_dir/<model_name>/<checkpoint_index>/. By default it sets the root directory "
        "to 3 levels above the fingerprints_dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. If not provided, it will be created inside the output_root_dir.",
    )
    parser.add_argument(
        "--index_type",
        "-i",
        type=str,
        default="ivfpq",
        help="Index type must be one of {'L2', 'IVF', 'IVFPQ', 'IVFPQ-RR', "
        "'IVFPQ-ONDISK', HNSW'}",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=None,
        help="Max number of items for index training. Default is all the data.",
    )
    parser.add_argument(
        "--n_probe",
        type=int,
        default=40,
        help="Number of neighboring cells to visit during search. Default is 40.",
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=20,
        help="Top k search for each segment. Default is 20",
    )
    parser.add_argument(
        "--segment_duration",
        type=float,
        default=1.0,
        help="Fingerprint context duration in seconds. Default is 1.0 seconds. "
        "Only used for display purposes.",
    )
    parser.add_argument(
        "--hop_duration",
        type=float,
        default=0.5,
        help="Fingerprint generation rate in seconds. Default is 0.5 seconds."
        "Only used for display purposes.",
    )
    parser.add_argument(
        "--test_seq_len",
        type=str,
        default="1,3,5,9,19",
        help="Comma-separated sequence lengths to test. Default is '1,3,5,9,19', "
        "corresponding to sequence durations of 1s, 2s, 3s, 5s, and 10s with "
        " a 1s segment and 0.5s hop duration.",
    )
    parser.add_argument(
        "--delta_n",
        type=int,
        default=7,
        help="Number of segments difference per query sequence. With hop_dur 0.5 seconds, "
        "corresponds to 3.5 seconds.",
    )
    parser.add_argument(
        "--display_interval",
        "-dp",
        type=int,
        default=100,
        help="Display interval. Default is 100, which updates the table every 100 "
        "query sequences.",
    )
    parser.add_argument(
        "--no_gpu", action="store_true", help="Use this flag to use CPU only."
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(seed_tf=False)

    main(
        query_dir=args.query_dir,
        database_dir=args.database_dir,
        output_root_dir=args.output_root_dir,
        output_dir=args.output_dir,
        index_type=args.index_type,
        max_train=args.max_train,
        n_probe=args.n_probe,
        top_k=args.top_k,
        segment_duration=args.segment_duration,
        hop_duration=args.hop_duration,
        test_seq_len=args.test_seq_len,
        delta_n=args.delta_n,
        display_interval=args.display_interval,
        no_gpu=args.no_gpu,
    )

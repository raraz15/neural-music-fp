"""Segment/sequence-wise retrieval and evaluation: implementation based on FAISS.
This script works on pre-determined conditions: each query fingerprint is a sequence of
segments with a fixed length. From each query fingerprint, we determine a set of
start positions and extract multiple sequences of segments with pre-determined lengths.

It implements a two-step retrieval, where first it uses a FAISS index to find the
top-k approximantely most similar segments in the database, and then it calculates a more
accurate sequence level similarity score for each candidate sequence.

The ground truth is obtained by the file names, so if you want to use this script
for another dataset, you should keep this in mind."""

import os
import time
import csv
import argparse
from pathlib import Path

from typing import Optional

import numpy as np
import pandas as pd

from nmfp.lib_retrieval.database import get_faiss_index
from nmfp.utils import set_seed, get_track_id_filled, fmt_ci, compute_symmetric_ci


def find_query_fingerprints(
    query_dir: Path,
) -> tuple[list[Path], list[int | None], list[Path]]:
    """Find the query fingerprints and track paths.

    Parameters:
    -----------
    query_dir: str
        The directory containing the query fingerprints. It should contain
        "track_paths.txt" file.

    Returns:
    --------
    query_fp_paths: list[Path]
        Paths to the query fingerprints.
    q_chunk_boundaries: list[int]
        Paths to the query chunks' boundary indices in the full tracks.
    query_audio_paths: list[str]
        Paths to the full query tracks.
    """

    print("Loading the query fingerprints...")

    # Get the fingerprint paths
    query_fp_paths = sorted(query_dir.rglob("*.npy"))
    assert len(query_fp_paths) > 0, f"No query fingerprints found in {query_dir}"
    print(f"{len(query_fp_paths):,} .npy files found in \033[32m{query_dir}\033[0m")

    audio_bound = {}
    with open(query_dir / "queries.csv") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            q_file_stem = Path(row[0]).stem
            # The start idx of the query chunk in the full track for segment-level search
            if len(row) > 1:
                chunk_bound_start = int(row[1])
            else:
                chunk_bound_start = None
            audio_bound[q_file_stem] = {
                "audio_path": Path(row[0]),
                "chunk_bound_start": chunk_bound_start,
            }

    # Follow the same order as query_fp_paths
    query_audio_paths = [audio_bound[fp.stem]["audio_path"] for fp in query_fp_paths]
    query_chunk_boundaries = [
        audio_bound[fp.stem]["chunk_bound_start"] for fp in query_fp_paths
    ]

    return query_fp_paths, query_chunk_boundaries, query_audio_paths


def load_ground_truth(ground_truth_path: Path):
    """The file stems are used for evaluating."""

    ground_truth = {}
    with open(ground_truth_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            query_path = Path(row[0])
            reference_path = Path(row[1])
            ground_truth[query_path.stem] = reference_path.stem
    return ground_truth


def main(
    query_dir: Path,
    database_mm_path: Path,
    ground_truth_path: Path,
    output_root_dir: Optional[Path],
    output_dir: Optional[Path],
    top_k: int,
    test_seq_len: str,
    delta_n: int,
    segment_level_eval: bool,
    index_type: str,
    n_probe: int,
    no_gpu: bool,
) -> None:
    """For a detailed explanation of the arguments, please refer to the argparse
    section at the end of this script."""

    assert top_k > 0, f"top_k must be > 0: {top_k}"
    assert delta_n > 0, f"delta_n must be > 0: {delta_n}"

    # Assumes the query and database fingerprints share a parent
    fingerprints_dir = database_mm_path.parent.parent

    # Determine the output directory
    if output_dir is None:

        # If output_root_dir is not provided, set it to 4 levels above the fingerprints_dir
        if output_root_dir is None:
            output_root_dir = fingerprints_dir.parent.parent.parent.parent / "eval"

        if output_root_dir.name != "eval":
            output_root_dir /= "eval"

        epoch = fingerprints_dir.name
        model_name = fingerprints_dir.parent.name
        data_name = fingerprints_dir.parent.parent.name
        output_dir = output_root_dir / data_name / model_name / epoch

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: \033[32m{output_dir}\033[0m")

    # '1,3,5' --> [1, 3, 5] and sort
    test_seq_len = np.asarray(sorted(list(map(int, test_seq_len.split(",")))))
    assert np.all(test_seq_len > 0), f"test_seq_len must be > 0: {test_seq_len}"
    max_seq_len = test_seq_len[-1]

    query_fp_paths, query_chunk_boundaries, query_audio_paths = find_query_fingerprints(
        query_dir
    )
    display_interval = min(500, len(query_fp_paths) // 10)

    if ground_truth_path is None:
        # By default, the ground truth should be here
        ground_truth_path = query_audio_paths[0].parent.parent / "ground_truth.csv"
    assert ground_truth_path.exists(), f"{ground_truth_path} does not exist"
    ground_truth = load_ground_truth(ground_truth_path)

    index_dict = {
        "index_type": index_type,
        "n_probe": n_probe,
    }
    index, db_track_boundaries, db_audio_paths, db_recons_memmap = get_faiss_index(
        merged_emb_path=database_mm_path,
        index_dict=index_dict,
        gpu=not no_gpu,
    )

    id_db_audio_paths_dict = {get_track_id_filled(_p): _p for _p in db_audio_paths}

    """ Segment/sequence-level search & evaluation """

    top1_track, top1_exact, top1_near, top1_far, analysis = [], [], [], [], []
    start_time, total_search_time = time.monotonic(), 0

    _str = "||" + "&".join(f"{_len:^12}" for _len in test_seq_len) + "||"
    pretty_print_len = len(_str)
    print(f'\n{" Top-1 Track Hit Rate (%) ":=^{pretty_print_len}}')
    print(f'{" Query Sequence Length ":-^{pretty_print_len}}')
    print(_str)
    print("-" * pretty_print_len)

    # For each query audio file's fingerprints
    seq_counter = 0
    for i, (q_fp_path, q_chunk_bound, q_audio_path) in enumerate(
        zip(query_fp_paths, query_chunk_boundaries, query_audio_paths)
    ):
        # FMA id of the query fingerprint
        q_track_ID = Path(q_fp_path).stem

        # fname of the ground truth track (in FMA it is the ID)
        gt_track_ID = ground_truth[q_track_ID]

        # Get the path to the database audio file
        # NOTE: this can be get from the ground truth too
        gt_track_path = id_db_audio_paths_dict[gt_track_ID]

        # Load the fingerprints of the query audio to memory
        q_fp = np.load(q_fp_path)
        # Make sure each query sequence is shorter than the query chunk and is equally spaced
        assert (
            q_fp.shape[0] >= max_seq_len
        ), f"Query sequence length {q_fp.shape[0]} is shorter than the query chunk {max_seq_len}"
        assert (
            delta_n < q_fp.shape[0]
        ), f"delta_n must be less than the query chunk: {delta_n} < {q_fp.shape[0]}"
        sequence_start_indices = np.arange(0, len(q_fp) + 1 - max_seq_len, delta_n)

        # Get a sequence inside the query from each of the sequence_start_indices
        for j, seq_start_idx in enumerate(sequence_start_indices):
            # Initialize the lists for each sequence
            top1_track.append(np.zeros(len(test_seq_len)).astype(int))
            top1_exact.append(np.zeros(len(test_seq_len)).astype(int))
            top1_near.append(np.zeros(len(test_seq_len)).astype(int))
            top1_far.append(np.zeros(len(test_seq_len)).astype(int))

            # The current sequence starts from this idx in the full track
            if segment_level_eval and q_chunk_bound is not None:
                q_start_segment = q_chunk_bound + seq_start_idx
            else:
                q_start_segment = None

            # Get the query sequence. if seq_len=1, query is a single fingerprint.
            # Otherwise its a sequence of fingerprints.
            q = q_fp[seq_start_idx : seq_start_idx + max_seq_len, :]
            assert (
                len(q) == max_seq_len
            ), f"Query sequence length mismatch: {len(q)} != {max_seq_len}"

            # For each segment in the sequence q, get the top k most similar
            # segments from the database, independently.
            _, I = index.search(
                q, top_k
            )  # _: distance, I: result IDs matrix (len(q), top_k)

            """Is uses the continuity of music. If the ith query segment matches a
            jth segment in the database, the (i+1)th query segment should match the
            (j+1)th segment in the database. Using this assumption, create candidate 
            sequences in the database. For each candidate segment, get a same length 
            sequence starting at from the past and including the candidate segment. 
            Calculate the similarity of the query sequence and the candidate sequence 
            then average the similarity scores."""
            for offset in range(len(I)):
                I[offset, :] -= offset

            # For each sequence length
            for k, seq_len in enumerate(test_seq_len):

                # Get the results for the current sequence length
                I_k = I[:seq_len, :].reshape(-1)

                # Remove the negative indices and indices that are out of bounds
                I_k = I_k[(I_k >= 0) & (I_k + seq_len <= db_track_boundaries[-1, 0])]
                if len(I_k) == 0:
                    continue

                # Keep sequences that are within the track boundaries NOTE: this is not crucial
                start_track_indices = (
                    np.searchsorted(db_track_boundaries[:, 0], I_k, side="right") - 1
                )
                end_track_indices = (
                    np.searchsorted(
                        db_track_boundaries[:, 0], I_k + seq_len, side="right"
                    )
                    - 1
                )
                I_k = I_k[start_track_indices == end_track_indices]
                if len(I_k) == 0:
                    continue

                # Remove duplicate segments
                candidates = np.unique(I_k)

                # Calculate the average similarity score for each candidate sequence
                _scores = np.zeros(len(candidates))
                for ci, cid in enumerate(candidates):
                    # candidate_sequence = index.reconstruct_n(cid, (cid + seq_len))
                    candidate_sequence = db_recons_memmap[cid : cid + seq_len, :]
                    _scores[ci] = np.mean(np.diag(np.dot(q, candidate_sequence.T)))

                """ Evaluate the sequence-level search in terms of track hit """

                # Position of the top fingerprint inside the DB
                # Each candidate referes to the start segment of when
                # the query sequence was initiated
                _idx = np.argsort(-_scores)[0]
                score = _scores[_idx]

                # Starting idx of the top fingerprint inside the DB
                p_fp_idx = candidates[_idx]

                # Position of the predicted track inside the DB
                p_track_idx = (
                    np.searchsorted(db_track_boundaries[:, 0], p_fp_idx, side="right")
                    - 1
                )
                # Position of the top predicted segment inside the track
                p_start_segment = p_fp_idx - db_track_boundaries[p_track_idx, 0]

                # Top prediction
                p_track_path = db_audio_paths[p_track_idx]
                p_track_ID = p_track_path.stem

                # Check if the top predicted track is correct
                top1_track[-1][k] = int(gt_track_ID == p_track_ID)

                # Check if the top predicted segment is aligned in time
                if segment_level_eval and top1_track[-1][k] == 1:
                    top1_exact[-1][k] = int(p_start_segment == q_start_segment)
                    top1_near[-1][k] = int(abs(p_start_segment - q_start_segment) <= 1)
                    top1_far[-1][k] = 1  # If it is not exact or near, it is far

                analysis.append(
                    {
                        "query_audio_path": q_audio_path,
                        "query_start_segment": q_start_segment,  # inside the query track
                        "seq_start_idx": seq_start_idx,  # inside the query chunk
                        "query_chunk_bound": q_chunk_bound,  # inside the query track
                        "seq_len": seq_len,
                        "gt_track_path": gt_track_path,
                        "pred_track_path": p_track_path,
                        "pred_start_segment": p_start_segment,
                        "score": score,
                    }
                )

                seq_counter += 1

        if ((i + 1) % display_interval) == 0 or (
            i == len(query_fp_paths) - 1
            and j == len(sequence_start_indices) - 1
            and k == len(test_seq_len) - 1
        ):
            mean_track, me_track = compute_symmetric_ci(np.stack(top1_track, axis=0))

            interval_time = time.monotonic() - start_time
            print(
                "||"
                + "&".join(
                    "{:^12}".format(arg) for arg in fmt_ci(mean_track, me_track)
                ),
                end="",
            )
            print(f"|| {interval_time / display_interval :>4.3f} s/file ", end="")
            print(f"[{i+1:>{len(str(len(query_fp_paths)))}}/{len(query_fp_paths)}] ||")
            total_search_time += interval_time
            start_time = time.monotonic()
    del db_recons_memmap

    print("-" * (pretty_print_len + 30) + "\n")
    print(
        f"Queried {seq_counter:,} sequences in {time.strftime('%H:%M:%S', time.gmtime(total_search_time))}",
    )

    # Calculate the average hit rates and confidence intervals
    mean_track, me_track = compute_symmetric_ci(np.stack(top1_track, axis=0))
    if segment_level_eval:
        mean_exact, me_exact = compute_symmetric_ci(np.stack(top1_exact, axis=0))
        mean_near, me_near = compute_symmetric_ci(np.stack(top1_near, axis=0))
        mean_far, me_far = compute_symmetric_ci(np.stack(top1_far, axis=0))

    df = pd.DataFrame(analysis)
    df.to_csv(output_dir / "analysis.csv", index=False)

    track_hit_rate_path = output_dir / "track_hit_rate.txt"
    with open(track_hit_rate_path, "w") as out_f:
        out_f.write(" & ".join([f"{_len} s" for _len in test_seq_len]) + "\n")
        out_f.write(" & ".join(fmt_ci(mean_track, me_track)) + "\n")
    print("Track-level Top1-Hit Rates:")
    with open(track_hit_rate_path, "r") as in_f:
        print(in_f.read())

    if segment_level_eval:
        segment_hit_rate_path = output_dir / "segment_hit_rate.txt"
        with open(segment_hit_rate_path, "w") as out_f:
            out_f.write(" & ".join([f"{_len} s" for _len in test_seq_len]) + "\n")
            out_f.write("Exact:\t" + " & ".join(fmt_ci(mean_exact, me_exact)) + "\n")
            out_f.write(" Near:\t" + " & ".join(fmt_ci(mean_near, me_near)) + "\n")
            out_f.write("  Far:\t" + " & ".join(fmt_ci(mean_far, me_far)) + "\n")

        print("Segment-level Top1-Hit Rates:")
        with open(segment_hit_rate_path, "r") as in_f:
            print(in_f.read())

    print(f"Saved the scores to {output_dir}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "queries",
        type=Path,
        help="Path to the query directory.",
    )
    parser.add_argument(
        "database",
        type=Path,
        help="Path to the database memmap.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="""Path to the ground truth file to evaluate the search results. If not provided,
        will look next to the query audio.""",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. If not provided, it will be created inside the output-root-dir.",
    )
    parser.add_argument(
        "--output-root-dir",
        type=Path,
        default=None,
        help="""Output root directory. Inside the root directory, the results will be saved 
        in output-root-dir/<model_name>/<checkpoint_index>/. By default it sets the root 
        directory to 2 levels above the fingerprints_dir.""",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=20,
        help="Top k search for each segment. Default is 20",
    )
    parser.add_argument(
        "--test-seq-len",
        type=str,
        default="1,3,5,9,19",
        help="""Comma-separated sequence lengths to test. Default is '1,3,5,9,19', 
        corresponding to sequence durations of 1s, 2s, 3s, 5s, and 10s with a 1s 
        segment and 0.5s hop duration.""",
    )
    parser.add_argument(
        "--delta-n",
        type=int,
        default=7,
        help="""Determines the number of segments to skip between the starting indices 
        of consecutive query sequences. This will be applied to each query file. Starting 
        at each starting index, multiple query sequences of length test_seq_len will be extracted.
        With hop_dur 0.5 seconds, corresponds to 3.5 seconds.""",
    )
    parser.add_argument(
        "--no-segment-level-eval",
        action="store_true",
        help="Use this flag to disable the segment-level search.",
    )
    parser.add_argument(
        "--index-type",
        "-i",
        type=str,
        default="ivfpq",
        help="Index type must be one of {'L2', 'IVFPQ'}",
    )
    parser.add_argument(
        "--n-probe",
        type=int,
        default=40,
        help="Number of neighboring cells to visit during search. Default is 40.",
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="Use this flag to use CPU only."
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(seed_tf=False)

    main(
        query_dir=args.queries,
        database_mm_path=args.database,
        ground_truth_path=args.ground_truth,
        output_root_dir=args.output_root_dir,
        output_dir=args.output_dir,
        index_type=args.index_type,
        n_probe=args.n_probe,
        top_k=args.top_k,
        test_seq_len=args.test_seq_len,
        delta_n=args.delta_n,
        no_gpu=args.no_gpu,
        segment_level_eval=not args.no_segment_level_eval,
    )

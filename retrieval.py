"""1 - For each segment in the query sequence, get the top k approximately most similar
segments from the database, independently.

2 - Create candidate sequences for each candidate segment in the database using the
continuity of music.

3 - For each candidate sequence, calculate the average similarity score between the
query sequence.
"""

import time
import csv
from pathlib import Path
import argparse

import numpy as np

from nmfp.utils import set_seed
from nmfp.lib_retrieval.database import get_faiss_index


def find_query_fingerprints(query_dir: Path) -> list[Path]:
    if query_dir.is_file():
        assert query_dir.suffix == ".npy", "Query file must be a .npy file"
        query_paths = [query_dir]
    elif query_dir.is_dir():
        print("Finding the query fingerprints...")
        query_paths = sorted(p for p in query_dir.rglob("*.npy"))
        print(
            f"{len(query_paths):,} query fingerprints found in \033[32m{query_dir}\033[0m"
        )
    else:
        raise ValueError(
            f"Invalid query: {query_dir}. It must be a file or a directory."
        )

    return query_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "queries",
        type=Path,
        help="Path to a single query or a directory with multiple queries.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for the results.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--database-embeddings",
        type=Path,
        default=None,
        help="Path to the database directory where embeddings are individual.",
    )
    group.add_argument(
        "--database-memmap",
        type=Path,
        default=None,
        help="Path to the database directory which had been built previously.",
    )
    group.add_argument(
        "--database-index",
        type=Path,
        default=None,
        help="Path to the directory that contains the trained and populated FAISS index.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Top k search for each segment. Default is 20",
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

    query_fp_paths = find_query_fingerprints(args.queries)
    display_interval = min(100, len(query_fp_paths) // 10)

    index_dict = {
        "index_type": args.index_type,
        "n_probe": args.n_probe,
    }
    index, db_track_boundaries, db_emb_paths, db_recons_memmap = get_faiss_index(
        index_path=args.database_index,
        merged_emb_path=args.database_memmap,
        embeddings_dir=args.database_embeddings,
        gpu=not args.no_gpu,
        index_dict=index_dict,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.csv"
    print(f"The results will be written to: \033[32m{output_path}\033[0m")
    with open(output_path, "w", newline="") as o_f:
        fieldnames = [
            "query_fp_path",
            "query_seq_len",
            "pred_track_path",
            "pred_start_segment",
            "score",
        ]
        writer = csv.DictWriter(o_f, fieldnames=fieldnames)
        writer.writeheader()

        print("-" * 25)
        start_time, total_search_time = time.monotonic(), 0
        for i, q_fp_path in enumerate(query_fp_paths):

            # Load the query fingerprints to memory
            q = np.load(q_fp_path)
            seq_len = len(q)

            # For each segment in the query sequence, get the top k approximately most similar
            # segments from the database, independently.
            _, I = index.search(
                q, args.top_k
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

            # Remove the negative indices and indices that are out of bounds
            I = I[(I >= 0) & (I + seq_len <= db_track_boundaries[-1, 0])]
            if len(I) == 0:
                print(
                    f"No candidates found for {q_fp_path} (sequence length = {seq_len})"
                )
                continue

            # Keep sequences that are within the track boundaries
            # NOTE: this is not crucial, if you have a huge database you might just
            # want to skip this step.
            start_track_indices = (
                np.searchsorted(db_track_boundaries[:, 0], I, side="right") - 1
            )
            end_track_indices = (
                np.searchsorted(db_track_boundaries[:, 0], I + seq_len, side="right")
                - 1
            )
            I = I[start_track_indices == end_track_indices]
            if len(I) == 0:
                print(
                    f"No candidates found for {q_fp_path} (sequence length = {seq_len})"
                )
                continue

            # Remove duplicates
            candidates = np.unique(I)

            # TODO add a flag here to disable the 2nd search
            # if db_recons_memmap is not None:

            # Calculate the average similarity score for each candidate sequence
            scores = np.zeros(len(candidates))
            for ci, cid in enumerate(candidates):
                # candidate_sequence = index.reconstruct_n(cid, (cid + seq_len))
                candidate_sequence = db_recons_memmap[cid : cid + seq_len, :]
                scores[ci] = np.mean(np.diag(np.dot(q, candidate_sequence.T)))

            # Get the metadata for each candidate and write to the output file
            for _idx in np.argsort(-scores):
                # Starting idx of the top fingerprint inside the DB
                p_fp_idx = candidates[_idx]
                # Position of the predicted track inside the DB
                p_track_idx = (
                    np.searchsorted(db_track_boundaries[:, 0], p_fp_idx, side="right")
                    - 1
                )
                # Position of the top predicted segment inside the track
                p_start_segment = p_fp_idx - db_track_boundaries[p_track_idx, 0]
                writer.writerow(
                    {
                        "query_fp_path": q_fp_path,
                        "query_seq_len": seq_len,
                        "pred_track_fname": db_emb_paths[p_track_idx],
                        "pred_start_segment": p_start_segment,
                        "score": scores[_idx],
                    }
                )

        if ((i + 1) % display_interval) == 0 or (i == len(query_fp_paths) - 1):
            interval_time = time.monotonic() - start_time
            print(
                f"||{interval_time / display_interval :>4.3f} s/query [{i+1}/{len(query_fp_paths)}]||"
            )
            total_search_time += interval_time
            start_time = time.monotonic()
    del db_recons_memmap
    print("-" * 25)
    print(
        f"All files queried in {time.strftime('%H:%M:%S', time.gmtime(total_search_time))}"
    )
    print("Done!")

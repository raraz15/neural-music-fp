"""Currently the database.csv contains filenames and not full paths.
Not sure if its good or bad."""

import sys
import time
from pathlib import Path
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


def get_shape(path: Path) -> tuple[str, int, int] | None:
    try:
        emb = np.load(path, mmap_mode="r")  # Avoids loading into RAM
        if emb.ndim != 2:
            return None
        return (path.name, emb.shape[0], emb.shape[1])
    except Exception:
        return None  # Could log path here if needed


def collect_all_shapes(emb_paths: list[Path], max_workers: int = 8):

    fnames_and_lens = {}
    emb_dim = None
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        # Submit one future per path
        future_to_path = {exe.submit(get_shape, p): p for p in emb_paths}

        for fut in as_completed(future_to_path):
            result = fut.result()
            if result is None:
                continue

            fname, n_rows, n_cols = result
            fnames_and_lens[fname] = n_rows

            # Ensure consistent embedding dimension
            if emb_dim is None:
                emb_dim = n_cols
            elif emb_dim != n_cols:
                raise ValueError(
                    f"Embedding-dimension mismatch: file {fname} has {n_cols}, "
                    f"but previous files had {emb_dim}"
                )

    # Sort by filename so that iteration order is deterministic
    fnames_and_lens = dict(sorted(fnames_and_lens.items()))
    assert emb_dim is not None, "No valid embedding files found."

    return fnames_and_lens, emb_dim


def _write_one_slice(in_path: Path, start: int, end: int, shared_mm: np.memmap) -> None:
    """
    Each thread does:
      1) mmap the input .npy (no full load),
      2) copy all rows into shared_mm[start:end, :],
      3) return without flushing.
    """
    in_mm = np.load(in_path, mmap_mode="r")
    shared_mm[start:end, :] = in_mm[:]  # slice‐assign in C (GIL released)
    del in_mm


def merge_embeddings_to_memmap(
    embeddings_dir: Path,
    flush_frequency: int = 100,  # How often to flush the memmap to disk
    max_workers: int = 8,
) -> tuple[Path, Path]:

    t0 = time.monotonic()
    print(
        "Creating a merged database from individual embeddings. This may take a while..."
    )

    assert (
        embeddings_dir.is_dir()
    ), f"Embeddings directory {embeddings_dir} does not exist."

    db_dir = embeddings_dir.parent
    db_mm_path = db_dir / "database.mm"
    db_csv_path = db_dir / "database.csv"

    if db_mm_path.exists():
        print(f"Database {db_mm_path} already exists.")
        user_input = input("Do you want to overwrite it? (y/n): ").strip().lower()
        if user_input == "y":
            print("Overwriting the database...")
            db_mm_path.unlink()
            db_csv_path.unlink(missing_ok=True)
        elif user_input == "n":
            if (
                input("Do you want to exit without creating the database? (y/n): ")
                .strip()
                .lower()
                == "y"
            ):
                print("Exiting without creating the database.")
                sys.exit(0)
            else:
                print("Continuing without overwriting the existing database.")
                return db_mm_path, db_csv_path
        else:
            print("Invalid input. Exiting without creating the database.")
            sys.exit(1)

    emb_paths = sorted(list(embeddings_dir.rglob("*.npy")))
    assert len(emb_paths) > 0, f"No embedding files found in {embeddings_dir}"
    print(f"Found {len(emb_paths):,} embedding files in {embeddings_dir}")

    print(f"Getting the embedding shapes with {max_workers} threads...")
    fnames_and_lens, emb_dim = collect_all_shapes(emb_paths, max_workers=max_workers)

    total_tracks = len(fnames_and_lens)
    assert total_tracks > 0, "No valid embedding files found."
    total_embeddings = sum(fnames_and_lens.values())
    assert total_embeddings > 0, "No valid embeddings found."
    db_shape = (total_embeddings, emb_dim)
    print(f"Total number of tracks in the database: {total_tracks:,}")
    print(f"Total number of embeddings in the database: {total_embeddings:,}")
    print(f"Embedding dim: {emb_dim:,}")

    print(f"Creating the merged database in {db_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    # Calculate the start and end indices for each file in the merged database
    starts_ends = []
    offset = 0
    for fname, n_rows in fnames_and_lens.items():
        start = offset
        end = offset + n_rows
        starts_ends.append((start, end))
        offset = end
    name_to_path = {p.name: p for p in emb_paths}

    memmap = np.memmap(db_mm_path, dtype=np.float32, mode="w+", shape=db_shape)
    memmap.flush()

    print(f"Merging embeddings with {max_workers} threads…")
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        for fname, (start, end) in zip(fnames_and_lens.keys(), starts_ends):
            in_path = name_to_path[fname]
            futures.append(exe.submit(_write_one_slice, in_path, start, end, memmap))
        done = 0
        total_files = len(futures)
        for fut in as_completed(futures):
            fut.result()  # re‐raise if that thread threw
            done += 1
            if done % flush_frequency == 0 or done == total_files:
                memmap.flush()
            if done % 10000 == 0 or done == total_files:
                print(f"{done} / {total_files} files merged")
    memmap.flush()
    del memmap

    with open(db_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # emb dim is redundant but simplifies the loading process
        writer.writerow(["file_name", "start", "end", "emb_dim"])
        for fname, (start, end) in zip(fnames_and_lens.keys(), starts_ends):
            writer.writerow([fname, start, end, emb_dim])
    print(f"Database CSV created at {db_csv_path}")

    print(
        f"Total elapsed time: {time.strftime('%H:%M:%S', time.gmtime(time.monotonic() - t0))}"
    )
    print(f"Database created at {db_mm_path}")

    return db_mm_path, db_csv_path

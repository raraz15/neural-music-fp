import csv
from pathlib import Path

import numpy as np

from .index import load_faiss_index, build_faiss_index_and_train
from .merge_embeddings import merge_embeddings_to_memmap


def load_database_metadata(database_dir: Path) -> tuple[np.ndarray, list[Path], int]:
    assert database_dir.is_dir(), f"Database directory {database_dir} does not exist."
    db_csv_path = database_dir / "database.csv"
    assert db_csv_path.exists(), f"Database CSV {db_csv_path} does not exist."

    # Load the CSV file to get the paths and boundaries
    audio_paths, track_boundaries = [], []
    with open(db_csv_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            audio_paths.append(Path(row[0]))
            track_boundaries.append((int(row[1]), int(row[2])))
    emb_dim = int(row[3])

    track_boundaries = np.array(track_boundaries)

    return track_boundaries, audio_paths, emb_dim


def load_database_memmap(db_mm_path: Path) -> tuple[np.memmap, np.ndarray, list[Path]]:

    print("Using existing database...")
    assert db_mm_path.exists(), f"Database {db_mm_path} does not exist."
    assert db_mm_path.suffix == ".mm", f"Database {db_mm_path} is not a memmap file."

    track_boundaries, audio_paths, emb_dim = load_database_metadata(db_mm_path.parent)

    database = np.memmap(
        db_mm_path, dtype=np.float32, mode="r", shape=(track_boundaries[-1, 1], emb_dim)
    )

    print(f"Loaded database from {db_mm_path}")
    print(f"Number of embeddings: {track_boundaries[-1, 1]:,}")
    print(f"Number of tracks: {len(track_boundaries):,}")

    return database, track_boundaries, audio_paths


def get_faiss_index(
    index_path: Path = None,
    merged_emb_path: Path = None,
    embeddings_dir: Path = None,
    gpu: bool = True,
    index_dict: dict = None,
) -> tuple:
    """----------------------------------------------------------------------

    • Calculation of sequence-level matching score requires reconstruction of
      vectors from FAISS index.
    • Unfortunately, current faiss.index.reconstruct_n(id_start, id_stop)
      supports only CPU index.
    • To enable candidate sequence reconstruction, we use the memmap file
    of the merged embeddings

    ----------------------------------------------------------------------"""

    if index_path is not None:
        index = load_faiss_index(index_path, gpu=gpu)
        track_boundaries, audio_paths, _ = load_database_metadata(index_path.parent)
        # We search for the reconstruction memmap in the same directory as the index
        mm_path = index_path.parent / "database.mm"
        mm, _, _ = load_database_memmap(mm_path)
    else:
        if merged_emb_path is not None:
            mm, track_boundaries, audio_paths = load_database_memmap(merged_emb_path)
            index_save_dir = merged_emb_path.parent
        else:
            mm_path, _ = merge_embeddings_to_memmap(embeddings_dir)
            mm, track_boundaries, audio_paths = load_database_memmap(mm_path)
            index_save_dir = mm_path.parent
        index = build_faiss_index_and_train(
            train_data=mm,
            gpu=gpu,
            index_dir=index_save_dir,
            **index_dict,
        )
    index.nprobe = index_dict["n_probe"]

    return index, track_boundaries, audio_paths, mm

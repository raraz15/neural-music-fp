import os
import time
from pathlib import Path

import numpy as np


def parse_memmap(
    mm_path: Path,
    mm_shape: tuple,
    track_paths: list[Path],
    track_boundaries: list | np.ndarray,
    output_dir: Path = None,
    delete_original: bool = False,
):

    start_time = time.monotonic()

    # Load the memmap file with read-only mode
    arr = np.memmap(mm_path, dtype="float32", mode="r", shape=mm_shape)

    if output_dir is None:
        output_dir = Path(mm_path).parent / "parsed"

    # Parse the fingerprints
    print(f"=== Parsing the mixture memmap to individual files ===")
    for i, (track_path, boundary) in enumerate(zip(track_paths, track_boundaries)):
        # Get the track id
        track_id = Path(track_path).stem
        # Group fingerprints by id[:3]
        track_dir = output_dir / track_id[:3]
        track_dir.mkdir(parents=True, exist_ok=True)
        # Load the fingerprints of the track from the memmap
        fp = arr[boundary[0] : boundary[1], :]
        # Convert to numpy array
        fp = np.array(fp)
        # Save query fingerprints
        fp_path = track_dir / f"{track_id}.npy"
        np.save(fp_path, fp)
        if i % 1000 == 0 or i == len(track_paths) - 1:
            print(f"Processed {i + 1:,} / {len(track_paths):,} files")

    # Close memmap
    del arr

    # Print the elapsed time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.monotonic() - start_time))
    print(f"Elapsed time for parsing the fingerprints: {elapsed_time}")

    # Remove the original memmap file if specified
    if delete_original:
        print("Deleting the original memmap file...")
        os.remove(mm_path)

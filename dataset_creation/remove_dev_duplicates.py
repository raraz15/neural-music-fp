"""This script reads the fma-mp3_hashes.json files that is created with 
exact_duplicate_finder.py, which is located in the same directory as this 
script. It removes the duplicate tracks from the train and validation sets 
of the NAFP dataset."""

import os
import sys
import argparse
import json
from glob import glob

code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(code_dir)

from nmfp.utils import get_track_id, track_id_to_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("nafp_dir", type=str, help="Path to the NAFP dataset.")
    parser.add_argument(
        "fma_dir", type=str, help="Path to the FMA-wav_8khz_16bit dataset."
    )
    args = parser.parse_args()

    """ Get the track ids of the NAFP dataset"""
    nafp_train_dir = os.path.join(
        args.nafp_dir, "music", "train-10k-30s", "fma_small_8k_plus_medium_2k"
    )
    nafp_val_dir = os.path.join(args.nafp_dir, "music", "val-query-db-500-30s", "db")

    train_fps = sorted(glob(os.path.join(nafp_train_dir, "**/*.wav"), recursive=True))
    val_fps = sorted(glob(os.path.join(nafp_val_dir, "**/*.wav"), recursive=True))

    train_ids = set([get_track_id(_path) for _path in train_fps])
    val_ids = set([get_track_id(_path) for _path in val_fps])

    # Write the original train and val ids to a text file
    with open(os.path.join(args.nafp_dir, "train_ids.txt"), "w") as out_f:
        for _id in train_ids:
            out_f.write(f"{_id}\n")
    with open(os.path.join(args.nafp_dir, "val_ids.txt"), "w") as out_f:
        for _id in val_ids:
            out_f.write(f"{_id}\n")

    """ Get the duplicates """

    # Read the hashes
    fma_mp3_hashes_path = os.path.join(args.fma_dir, "fma-mp3_hashes.json")
    with open(fma_mp3_hashes_path, "r") as in_f:
        hashes = json.load(in_f)

    # Filter the cliques with more than one track
    bitwise_duplicates = {k: v for k, v in hashes.items() if len(v) > 1}

    # Convert to track ids
    bitwise_duplicates = [
        set([get_track_id(_v) for _v in v]) for v in bitwise_duplicates.values()
    ]
    print("Number of total duplicate cliques: ", len(bitwise_duplicates))
    print(
        f"{sum([len(v) for v in bitwise_duplicates])} tracks are in duplicate cliques."
    )

    """ Remove the duplicates from the NAFP train and dev sets """

    # Find which cliques contain tracks from the NAFP train and dev sets
    train_remove_ids = set()
    val_remove_ids = set()
    for clique in bitwise_duplicates:
        train_intersection = train_ids.intersection(clique)
        if len(train_intersection) > 1:
            # Remove an arbitrary track id from the intersection
            train_intersection = set(list(train_intersection)[1:])
            train_remove_ids = train_remove_ids.union(train_intersection)
            print(f"{train_intersection} are removed from train_ids")
        val_intersection = val_ids.intersection(clique)
        if len(val_intersection) > 1:
            # Remove an arbitrary track id from the intersection
            val_intersection = set(list(val_intersection)[1:])
            val_remove_ids = val_remove_ids.union(val_intersection)
            print(f"{val_intersection} are removed from val_ids")

    # Remove the tracks from the NAFP train and dev sets
    if len(train_remove_ids) > 0:
        train_duplicates_dir = os.path.join(
            args.nafp_dir,
            "music",
            "duplicates",
            "train-10k-30s",
            "fma_small_8k_plus_medium_2k",
        )
        print(f"Writing the removed tracks to {train_duplicates_dir}")
        os.makedirs(train_duplicates_dir, exist_ok=True)
        for _id in train_remove_ids:
            in_path = track_id_to_path(_id, nafp_train_dir)
            out_path = track_id_to_path(_id, train_duplicates_dir)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            print(f"Moving {in_path} to {out_path}")
            os.rename(in_path, out_path)

    if len(val_remove_ids) > 0:
        val_duplicates_dir = os.path.join(
            args.nafp_dir, "music", "duplicates", "val-query-db-500-30s", "db"
        )
        print(f"Writing the removed tracks to {val_duplicates_dir}")
        os.makedirs(val_duplicates_dir, exist_ok=True)
        for _id in val_remove_ids:
            in_path = track_id_to_path(_id, nafp_val_dir)
            out_path = track_id_to_path(_id, val_duplicates_dir)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            print(f"Moving {in_path} to {out_path}")
            os.rename(in_path, out_path)

    print("Done!")

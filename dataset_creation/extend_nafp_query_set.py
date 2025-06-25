"""This script extends the NAFP query set in a few steps. Frist it looks at the NAFP
dataset to exlcude the development set (train + val). Then it loads the FMA dataset
tracks.csv file to get more information. Finally, it eliminates the bitwise exact
duplicates of the FMA dataset. From the remaining tracks that are longer than 30.25
seconds, it samples a query set of 9500 tracks. No stratification is applied during
the sampling. The original 500 query tracks of NAFP is added to this query set."""

import os
import argparse
import sys
import shutil
import json
from glob import glob

code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(code_dir)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nmfp.audio_processing import check_wav_file
from nmfp.utils import (
    set_seed,
    SEED,
    get_track_id,
    track_id_to_path,
    load_fma_tracks_csv,
)

SAMPLE_RATE = 8000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("nafp_dir", type=str, help="Path to the NAFP dataset.")
    parser.add_argument(
        "fma_dir", type=str, help="Path to the FMA-wav_8khz_16bit dataset."
    )
    parser.add_argument("output_dir", type=str, help="Output directory.")
    parser.add_argument(
        "--N_test", type=int, default=10000, help="Number of total test tracks."
    )
    parser.add_argument(
        "--chunk_duration",
        type=float,
        default=30.0,
        help="Duration of a sampled audio chunk in seconds.",
    )
    parser.add_argument(
        "--max_shift_duration",
        type=float,
        default=0.25,
        help="The maximum amount of time shift that will be applied "
        "to the clean chunk in seconds. This is done to simulate the "
        "real-life mismatch between the query moment and stored "
        "fingerprints.",
    )

    args = parser.parse_args()

    # Set the seed
    set_seed(seed_tf=False)

    # Get the necessary paths
    nafp_train_dir = os.path.join(args.nafp_dir, "music", "train-10k-30s")
    nafp_val_dir = os.path.join(args.nafp_dir, "music", "val-query-db-500-30s")
    nafp_query_dir = os.path.join(args.nafp_dir, "music", "test-query-db-500-30s", "db")

    tracks_csv_path = os.path.join(args.fma_dir, "tracks.csv")
    fma_mp3_hashes_path = os.path.join(args.fma_dir, "fma-mp3_hashes.json")
    fma_audio_dir = os.path.join(args.fma_dir, "audio")

    """ Load the tracks.csv """

    df = load_fma_tracks_csv(tracks_csv_path)
    print(f"{df.shape[0]:,} tracks are in the FMA dataset originally.")

    """ Remove duplicates """

    # Read the hashes
    with open(fma_mp3_hashes_path, "r") as in_f:
        hashes = json.load(in_f)

    # Filter the cliques with more than one track
    bitwise_duplicates = {k: v for k, v in hashes.items() if len(v) > 1}

    # Convert to track ids
    bitwise_duplicates = [
        [get_track_id(_v) for _v in v] for v in bitwise_duplicates.values()
    ]
    print("Number of duplicate cliques: ", len(bitwise_duplicates))
    print(
        f"{sum([len(v) for v in bitwise_duplicates]):,} tracks are in duplicate cliques."
    )

    # Keep only the first track id for each duplicate clique
    print("Removing the duplicates from the FMA dataset.")
    for clique in bitwise_duplicates:
        df.drop(clique[1:], inplace=True)
    print(f"{df.shape[0]:,} tracks are remaining after removing duplicates.")

    """ Output directory """

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Outputs will be written to {args.output_dir}")

    # Save this script next to the output directory
    print(f"Copying this script to {args.output_dir}")
    shutil.copy(os.path.realpath(__file__), args.output_dir)

    """Information About NAFP Development Set and Query Set"""

    train_fps = sorted(
        glob(os.path.join(nafp_train_dir, "**", "*.wav"), recursive=True)
    )
    val_fps = sorted(glob(os.path.join(nafp_val_dir, "**", "*.wav"), recursive=True))
    query_fps = sorted(
        glob(os.path.join(nafp_query_dir, "**", "*.wav"), recursive=True)
    )

    train_ids = set([get_track_id(_path) for _path in train_fps])
    val_ids = set([get_track_id(_path) for _path in val_fps])
    dev_ids = train_ids.union(val_ids)
    query_ids = set([get_track_id(_path) for _path in query_fps])

    # Find out which query ids are in the database after duplicate removal
    query_ids = query_ids.intersection(set(df.index.to_list()))
    N_old_query = len(query_ids)
    print(f"{N_old_query} query tracks are in the database after duplicate removal.")

    # We will put all the non_dev tracks to the database
    df = df.loc[~df.index.isin(dev_ids)]
    print(f"{df.shape[0]:,} tracks will be used for the database.")

    # Write the true audio durations and the paths
    print("Reading the true audio durations and the paths.")
    for track_id in df.index.to_list():
        track_path = track_id_to_path(track_id, audio_dir=fma_audio_dir)
        size = check_wav_file(track_path, SAMPLE_RATE)
        df.loc[track_id, ("track", "duration_read")] = np.round(size / SAMPLE_RATE, 2)

    # Create the directory of the database
    database_dir = os.path.join(args.output_dir, "database")
    os.makedirs(database_dir, exist_ok=True)

    # Move the database tracks from the FMA dataset to the database directory
    # We move instead of copy to save disk space
    print("Moving the database tracks to the output directory.")
    for track_id in df.index.to_list():
        # Create the new directory
        track_dir = os.path.join(database_dir, track_id[:3])
        os.makedirs(track_dir, exist_ok=True)
        # Path of the track
        track_path = track_id_to_path(track_id, audio_dir=fma_audio_dir)
        # Move the track
        shutil.move(track_path, track_dir)

    # Write the database
    database_path = os.path.join(database_dir, "test_database.csv")
    print(f"Writing the database to {database_path}")
    df.to_csv(database_path)

    """ Duration based filtering """

    # Split the df based on audio length
    t_min = args.chunk_duration + args.max_shift_duration
    df = df[df[("track", "duration_read")] >= t_min]
    print(f"{df.shape[0]:,} tracks are longer than {t_min} seconds.")

    # Get their track ids
    long_track_ids = set(df.index.to_list())

    """ Remove the development set and the old query set from the long tracks """

    # We will sample queries from this dataset. Remove query ids because we will use them
    interested_ids = list(long_track_ids.difference(query_ids))
    _df = df.loc[interested_ids]
    print(
        f"{_df.shape[0]:,} tracks will be used for sampling after removing the old query set."
    )
    assert len(set(_df.index.to_list()).intersection(dev_ids)) == 0
    assert len(set(_df.index.to_list()).intersection(query_ids)) == 0
    assert len(df) >= args.N_test, "Not enough tracks left for sampling."

    """ Sample from Long, Non-development Tracks """

    _, _df = train_test_split(
        _df,
        test_size=args.N_test - N_old_query,  # Will use the old query ids later
        random_state=SEED,
        shuffle=True,
    )
    assert (
        len(_df) == args.N_test - N_old_query
    ), "Something went wrong with the sampling."

    # Add the old test_query_db
    print("Concatenate the old query set to the new query set.")
    df = pd.concat([df.loc[list(query_ids)], _df])
    assert len(df) == args.N_test, "Something went wrong with the concatenation."
    print(f"{len(df):,} tracks are used for the new query set.")

    # Create the directory of the query set
    query_dir = os.path.join(args.output_dir, "queries")
    os.makedirs(query_dir, exist_ok=True)

    # Write the query set
    query_path = os.path.join(query_dir, "test_queries.csv")
    print(f"Writing the query set to {query_path}")
    df.to_csv(query_path)

    print("Done!")

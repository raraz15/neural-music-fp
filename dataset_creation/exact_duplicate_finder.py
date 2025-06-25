"""This script is used to find exact duplicates of mp3 files in a directory.
Adapted from https://github.com/mdeff/fma/issues/23"""

import os
import glob
import json
import argparse
import hashlib
from joblib import Parallel, delayed


def hash_one(fname):
    hsh = hashlib.sha384()
    hsh.update(open(fname, "rb").read())
    return hsh.digest().hex()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "mp3_dir", type=str, help="Path to directory that contains mp3 files."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to directory where hashes will be saved.",
    )
    args = parser.parse_args()

    fnames = glob.glob(os.path.join(args.mp3_dir, "**", "*.mp3"), recursive=True)
    print("Found {} mp3 files.".format(len(fnames)))

    pool = Parallel(n_jobs=-2, verbose=20)
    dfx = delayed(hash_one)
    fhashes = pool(dfx(fn) for fn in fnames)

    groups = dict()
    for fh, fn in zip(fhashes, fnames):
        if fh not in groups:
            groups[fh] = []
        groups[fh].append(os.path.splitext(os.path.basename(fn))[0])

    # Save the hashes one directory up
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "fma-mp3_hashes.json")
    print(f"Saving hashes to {output_path}")
    with open(output_path, "w") as out_f:
        json.dump(groups, out_f, indent=4)

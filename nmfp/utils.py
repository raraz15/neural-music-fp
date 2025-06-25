import os
import yaml
import ast
import random

import pandas as pd
import numpy as np

SEED = 27  # License plate code of Gaziantep, gastronomical capital of Türkiye


def set_seed(seed: int = SEED, seed_tf: bool = True) -> None:

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if seed_tf:
        import tensorflow as tf

        tf.random.set_seed(seed)
    print(f"Random seed set as {seed}")


def load_config(config_filepath: str) -> dict:
    """Load the model and trainig configuration from a YAML file."""

    if os.path.splitext(config_filepath)[1] == ".yaml":
        if os.path.exists(config_filepath):
            print(f"Configuration from {config_filepath}")
            with open(config_filepath, "r") as f:
                cfg = yaml.safe_load(f)
            return cfg
        else:
            raise FileNotFoundError(
                f"ERROR! Configuration file {config_filepath} is missing!!"
            )
    else:
        raise ValueError(
            f"ERROR! Configuration file {config_filepath} is not a YAML file!!"
        )


def print_config(cfg: dict) -> None:
    os.system("")
    print("\033[36m" + yaml.dump(cfg, indent=4, width=120, sort_keys=False) + "\033[0m")


def get_track_id_filled(_path: str) -> str:
    """Get the track id from the path which is filleed with zeros.
    Works for FMA dataset."""

    return os.path.splitext(os.path.basename(_path))[0]


def get_track_id(_path: str) -> int:
    """Get the track id from the path without the zeros as an integer.
    Works for the FMA dataset."""

    return int(os.path.basename(_path).split(".")[0].lstrip("0"))


def track_id_to_path(track_id: int, audio_dir: str) -> str:
    """Convert the integer track id to the path of the audio file.
    Works for the FMA dataset."""

    _track_id = str(track_id).zfill(6)
    return os.path.join(audio_dir, _track_id[:3], f"{_track_id}.wav")


def load_fma_tracks_csv(filepath: str) -> pd.DataFrame:
    """Load the tracks.csv file of the FMA dataset. Adapted from
    https://github.com/mdeff/fma"""

    tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

    COLUMNS = [
        ("track", "tags"),
        ("album", "tags"),
        ("artist", "tags"),
        ("track", "genres"),
        ("track", "genres_all"),
    ]
    for column in COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)

    COLUMNS = [
        ("track", "date_created"),
        ("track", "date_recorded"),
        ("album", "date_created"),
        ("album", "date_released"),
        ("artist", "date_created"),
        ("artist", "active_year_begin"),
        ("artist", "active_year_end"),
    ]
    for column in COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])

    SUBSETS = ("small", "medium", "large")
    try:
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            "category", categories=SUBSETS, ordered=True
        )
    except (ValueError, TypeError):
        # the categories and ordered arguments were removed in pandas 0.25
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            pd.CategoricalDtype(categories=SUBSETS, ordered=True)
        )

    COLUMNS = [
        ("track", "genre_top"),
        ("track", "license"),
        ("album", "type"),
        ("album", "information"),
        ("artist", "bio"),
    ]
    for column in COLUMNS:
        tracks[column] = tracks[column].astype("category")

    return tracks


def compute_symmetric_ci(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute symmetric Wald confidence intervals for each column of a binary matrix.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_metrics)
        Binary (0/1) observations for each metric.

    Returns
    -------
    means : np.ndarray, shape (n_metrics,)
        Proportion of successes in each column (values in [0,1]).
    margins : np.ndarray, shape (n_metrics,)
        Symmetric margin of error so that the 95% CI is means ± margins.
    """

    assert len(data) > 0, "Data array is empty."

    n = data.shape[0]
    p = data.mean(axis=0)  # sample proportions
    se = np.sqrt(p * (1 - p) / n)  # standard error
    margin = 1.96 * se

    # convert to percentage and round
    p = np.round(100 * p, 1)
    margin = np.round(100 * margin, 1)

    return p, margin


def fmt_ci(
    means: np.ndarray, margins: np.ndarray, fmt: str = "{:.1f} ± {:.1f}"
) -> list[str]:
    return [fmt.format(m, e) for m, e in zip(means, margins)]

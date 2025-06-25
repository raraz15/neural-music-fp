"""This module contains the Dataset class for evaluating a model, which contains
the loaders for the database and query tracks. Audio degradation is not possible
 with EvaluationDataset dataloaders. For degradation, check the
 dataset_creation/augment_query_chunks.py script."""

import os
import glob

from nmfp.dataloaders.loaders import EvaluationLoader


class EvaluationDataset:
    """Build a dataset for fingerprinter evaluation.

    Public Methods
    --------------
        get_database_loader()
        get_query_loader()

    """

    def __init__(self, cfg: dict):
        """Initialize the EvaluationDataset object.

        Parameters
        ----------
            cfg : dict
                A dictionary containing model configurations.
        """

        self.cfg = cfg

        # Model parameters
        self.segment_duration = cfg["MODEL"]["AUDIO"]["SEGMENT_DUR"]
        self.fs = cfg["MODEL"]["AUDIO"]["FS"]
        self.stft_hop = cfg["MODEL"]["INPUT"]["STFT_HOP"]
        self.n_fft = cfg["MODEL"]["INPUT"]["STFT_WIN"]
        self.n_mels = cfg["MODEL"]["INPUT"]["N_MELS"]
        self.fmin = cfg["MODEL"]["INPUT"]["F_MIN"]
        self.fmax = cfg["MODEL"]["INPUT"]["F_MAX"]
        self.dynamic_range = cfg["MODEL"]["INPUT"]["DYNAMIC_RANGE"]
        self.scale_inputs = cfg["MODEL"]["INPUT"]["SCALE"]

        print("Initialized the evaluation dataset.")

    def get_database_loader(
        self, db_tracks: str, batch_size: int, hop_duration: float
    ) -> EvaluationLoader:
        """This loader will load the database tracks and segment them according
        to the model configurations in the config file.

        Parameters
        ----------
            db_tracks : str
                A directory containing the database tracks
            batch_size : int
                The batch size for the dataloader
            hop_duration : float
                The hop duration for the dataloader

        Returns
        -------
            loader : EvaluationLoader
                The dataloader of the tracks.

        """

        print("Creating the database loader...")

        # Find the database tracks
        if os.path.isdir(db_tracks):
            db_track_paths = sorted(
                glob.glob(os.path.join(db_tracks, "**", "*.wav"), recursive=True)
            )
        elif os.path.isfile(db_tracks) and db_tracks.endswith(".txt"):
            with open(db_tracks, "r") as f:
                db_track_paths = [line.strip() for line in f]
        else:
            raise ValueError("Invalid database tracks specification.")
        assert len(db_track_paths) > 0, f"No audio files found in {db_tracks}"
        print(f"{len(db_track_paths):,} database audio files found.")

        # Create the database dataset
        return EvaluationLoader(
            track_paths=db_track_paths,
            segment_duration=self.segment_duration,
            hop_duration=hop_duration,
            fs=self.fs,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            dynamic_range=self.dynamic_range,
            scale_output=self.scale_inputs,
            bsz=batch_size,
        )

    def get_query_loader(
        self, query_chunks: str, batch_size: int, hop_duration: float
    ) -> EvaluationLoader:
        """This loader will load the query chunks and segment them according to the model
        configuration in the config file. We use chunks of audio with the same duration for
        creating the query dataset, but the loader can work with full tracks or chunks with
        different durations too. They should have been degraded previously.

        Parameters
        ----------
            query_chunks : str
                Directory containing the query chunks or a line delimited text
                file containing paths.
            batch_size : int
                The batch size for the dataloader
            hop_duration : float
                The hop duration for the dataloader

        Returns
        -------
            loader : EvaluationLoader
                The dataloader of the query chunks.
        """

        print("Creating the query loader...")

        # Find the query audio files
        if os.path.isdir(query_chunks):
            self.query_chunk_paths = sorted(
                glob.glob(os.path.join(query_chunks, "**", "*.wav"), recursive=True)
            )
        elif os.path.isfile(query_chunks) and query_chunks.endswith(".txt"):
            with open(query_chunks, "r") as f:
                self.query_chunk_paths = [line.strip() for line in f]
        else:
            raise ValueError("Invalid query chunks specification.")
        assert (
            len(self.query_chunk_paths) > 0
        ), f"No audio files found in {query_chunks}"
        print(f"{len(self.query_chunk_paths):,} audio files found.")

        # Find the boundary files for segment-level evaluation and listening
        self.query_boundary_paths = []
        for chunk_path in self.query_chunk_paths:
            boundary_path = chunk_path.replace(".wav", ".npy")
            if os.path.exists(boundary_path):
                self.query_boundary_paths.append(boundary_path)
        if len(self.query_boundary_paths) == 0:
            print(
                "No boundary files found for the query chunks. "
                "Evaluation will be performed at the track level only."
            )

        # Create the query dataset
        ds = EvaluationLoader(
            track_paths=self.query_chunk_paths,
            segment_duration=self.segment_duration,
            hop_duration=hop_duration,
            fs=self.fs,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            dynamic_range=self.dynamic_range,
            scale_output=self.scale_inputs,
            bsz=batch_size,
        )

        return ds

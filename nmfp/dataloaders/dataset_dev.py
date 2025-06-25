"""Dataset class for the training set."""

import os
import glob

from nmfp.dataloaders.loaders import TrainLoader


class DevelopmentDataset:
    """Build a dataset for training a fingerprinter.

    Public Methods
    --------------
        get_train_loader()

    """

    def __init__(self, cfg: dict):
        """Initialize the DevelopmentDataset object.

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

        """Training Parameters"""
        self.tr_audio_dir = cfg["TRAIN"]["MUSIC_DIR"]

        self.tr_n_anchors = cfg["TRAIN"]["N_ANCHORS"]
        self.tr_n_positives_per_anchor = cfg["TRAIN"]["N_POSITIVES_PER_ANCHOR"]

        self.tr_hop_duration = cfg["TRAIN"]["AUDIO"]["SEGMENT_HOP_DUR"]
        self.tr_past_context_duration = cfg["TRAIN"]["AUDIO"]["PAST_CONTEXT_DUR"]
        self.tr_max_offset_dur_anch = cfg["TRAIN"]["AUDIO"]["MAX_OFFSET_DUR_ANCHOR"]
        self.tr_max_offset_dur_pos = cfg["TRAIN"]["AUDIO"]["MAX_OFFSET_DUR_POS"]
        self.tr_chunk_duration = cfg["TRAIN"]["AUDIO"]["CHUNK_DUR"]

        # Background noise degradation parameters
        self.tr_use_bg_deg = cfg["TRAIN"]["DEGRADATION"]["TD"]["BG"]
        self.tr_bg_root_dir = cfg["TRAIN"]["DEGRADATION"]["TD"]["BG_ROOT"]
        self.tr_bg_snr = cfg["TRAIN"]["DEGRADATION"]["TD"]["BG_SNR"]
        self.tr_bg_paths = []

        # Room IR degradation parameters
        self.tr_use_rir_deg = cfg["TRAIN"]["DEGRADATION"]["TD"]["RIR"]
        self.tr_rir_root_dir = cfg["TRAIN"]["DEGRADATION"]["TD"]["RIR_ROOT"]
        self.tr_pre_rir_amp_range = cfg["TRAIN"]["DEGRADATION"]["TD"][
            "PRE_RIR_AMP_RANGE"
        ]
        self.tr_rir_paths = []

        # Microphone degradation parameters
        self.tr_mir_root_dir = cfg["TRAIN"]["DEGRADATION"]["TD"]["MIR_ROOT"]
        self.tr_use_mir_deg = cfg["TRAIN"]["DEGRADATION"]["TD"]["MIR"]
        self.tr_pre_mir_amp_range = cfg["TRAIN"]["DEGRADATION"]["TD"][
            "PRE_MIR_AMP_RANGE"
        ]
        self.tr_mir_paths = []

    def get_train_loader(self, reduce_items_p: float = 100) -> TrainLoader:
        """Audio loader for the training data. You should use audio chunks with the same
        length for training. Audio degradations are possible. The folder structure should
        be as follows:
            self.tr_audio_dir/
                dir0/
                    track1.wav
                    ...
                dir1/
                    track1.wav
                    ...
                ...

        Parameters
        ----------
            reduce_items_p : float (default 100)
                Reduce the number of items in each track to this percentage.

        Returns
        -------
            loader : TrainLoader
                The dataloader of the training audio files.

        """

        print("Creating the training dataset...")

        # Find the wav tracks
        self.tr_chunk_paths = sorted(
            glob.glob(os.path.join(self.tr_audio_dir, "**/*.wav"), recursive=True)
        )
        assert (
            len(self.tr_chunk_paths) > 0
        ), f"No audio files found in {self.tr_audio_dir}"
        print(f"{len(self.tr_chunk_paths):,} audio files found.")

        # Reduce the total number of tracks if requested
        if reduce_items_p < 100:
            print(f"Reducing the number of audio files used to {reduce_items_p}%.")
            self.tr_chunk_paths = self.tr_chunk_paths[
                : int(len(self.tr_chunk_paths) * reduce_items_p / 100)
            ]
            print(f"Reduced to {len(self.tr_chunk_paths):,} audio files.")

        # Find the audio degradation files
        self._read_train_degradations()

        # Create the loader
        loader = TrainLoader(
            chunk_paths=self.tr_chunk_paths,
            segment_duration=self.segment_duration,
            hop_duration=self.tr_hop_duration,
            chunk_duration=self.tr_chunk_duration,
            past_context_duration=self.tr_past_context_duration,
            fs=self.fs,
            n_fft=self.n_fft,
            stft_hop=self.stft_hop,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            dynamic_range=self.dynamic_range,
            scale_output=self.scale_inputs,
            n_anchors=self.tr_n_anchors,
            n_ppa=self.tr_n_positives_per_anchor,
            shuffle=True,
            max_offset_dur_anch=self.tr_max_offset_dur_anch,
            max_offset_dur_pos=self.tr_max_offset_dur_pos,
            bg_deg_parameters=self.tr_bg_parameters,
            room_ir_deg_parameters=self.tr_rir_parameters,
            mic_ir_deg_parameters=self.tr_mir_parameters,
        )

        return loader

    def _read_train_degradations(self):
        """Read the audio degradation files of the training set."""

        if self.tr_use_bg_deg:
            self.tr_bg_paths = sorted(
                glob.glob(os.path.join(self.tr_bg_root_dir, "**/*.wav"), recursive=True)
            )
            print(f" #BG clips: {len(self.tr_bg_paths):>6,}")
            assert (
                len(self.tr_bg_paths) > 0
            ), f"No background noise clips found in {self.tr_bg_root_dir}"
        self.tr_bg_parameters = [
            self.tr_use_bg_deg,
            self.tr_bg_paths,
            self.tr_bg_snr,
        ]

        if self.tr_use_rir_deg:
            self.tr_rir_paths = sorted(
                glob.glob(os.path.join(self.tr_rir_root_dir, "**/*.wav"), recursive=True)
            )
            print(f"#RIR clips: {len(self.tr_rir_paths):>6,}")
            assert (
                len(self.tr_rir_paths) > 0
            ), f"No room impulse response found in {self.tr_rir_root_dir}"
        self.tr_rir_parameters = [
            self.tr_use_rir_deg,
            self.tr_rir_paths,
            self.tr_pre_rir_amp_range,
        ]

        if self.tr_use_mir_deg:
            self.tr_mir_paths = sorted(
                glob.glob(os.path.join(self.tr_mir_root_dir, "**/*.wav"), recursive=True)
            )
            print(f"#MIR clips: {len(self.tr_mir_paths):>6,}")
            assert (
                len(self.tr_mir_paths) > 0
            ), f"No microphone impulse response found in {self.tr_mir_root_dir}"
        self.tr_mir_parameters = [
            self.tr_use_mir_deg,
            self.tr_mir_paths,
            self.tr_pre_mir_amp_range,
        ]

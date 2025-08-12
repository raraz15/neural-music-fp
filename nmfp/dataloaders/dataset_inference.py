from pathlib import Path

from nmfp.dataloaders.loaders import InferenceLoader


class InferenceDataset:

    def __init__(self, cfg: dict):

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

        self.supported_audio_ext = {".wav", ".flac", ".mp3", ".aac", ".ogg"}

        print("Initialized the inference dataset.")

    def get_loader(
        self, path_pairs: list[tuple[Path, Path]], hop_duration: float
    ) -> InferenceLoader:

        print("Creating the inference loader...")
        return InferenceLoader(
            path_pairs=path_pairs,
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
        )

    def find_audio_paths(self, inp: Path) -> list[Path]:
        print(f"Finding the audio files in {inp}")

        audio_paths = []

        if inp.is_file():
            if inp.suffix == ".txt":
                with inp.open("r") as f:
                    audio_paths = [Path(line.strip()) for line in f]
            elif inp.suffix in self.supported_audio_ext:
                audio_paths = [inp]
            else:
                raise ValueError("Invalid audio paths specification.")

        elif inp.is_dir():
            audio_paths = [
                p for ext in self.supported_audio_ext for p in inp.rglob(f"*{ext}")
            ]

        else:
            raise ValueError("Invalid audio paths specification.")

        audio_paths = sorted(audio_paths)

        assert len(audio_paths) > 0, f"No audio files found."
        print(f"{len(audio_paths):,} audio files found.")

        return audio_paths

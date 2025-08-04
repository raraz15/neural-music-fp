from pathlib import Path

import numpy as np
from tensorflow.keras.utils import Sequence
import essentia.standard as es

from nmfp import audio_processing


class InferenceLoader(Sequence):
    """Dataloader object for loading audio clips, segmenting, and
    extracting power melspectrograms. It can deal with various audio formats
    and is designed for inference purposes, where the audio files are processed
    one at a time. It does not support batching or shuffling."""

    def __init__(
        self,
        audio_paths: list[Path],
        segment_duration=1,
        hop_duration=0.5,
        fs=8000,
        n_fft=1024,
        stft_hop=256,
        n_mels=256,
        f_min=300,
        f_max=4000,
        dynamic_range=80,
        scale_output=True,
    ):
        """
        Parameters
        ----------
            audio_paths : list[Path],
                Aduio clip paths as a list.
            segment_duration : (float), optional
                Segment duration in seconds. The default is 1.
            hop_duration : (float), optional
                Hop-size of segments in seconds. The default is .5.
            fs : (float), optional
                Sampling rate. The default is 8000.
            n_fft: (int), optional
                FFT size. Default is 1024.
            stft_hop : (int), optional
                STFT hop-size. Default is 256.
            n_mels : (int), optional
                Number of mel-bands. Default is 256.
            f_min : (int), optional
                Minimum frequency of the mel-bands. Default is 300.
            f_max : (int), optional
                Maximum frequency of the mel-bands. Default is 4000.
            scale_output : (bool), optional
                Scale the power mel-spectrogram. The default is True.
        """

        super(InferenceLoader, self).__init__()

        # Check the input parameters
        assert segment_duration > 0, "segment_duration must be > 0"
        assert hop_duration > 0, "hop_duration must be > 0"
        assert (
            hop_duration <= segment_duration
        ), "hop_duration must be <= segment_duration"

        # Set the parameters
        self.segment_duration = segment_duration
        self.segment_length = int(fs * self.segment_duration)  # Convert to samples
        self.hop_duration = hop_duration
        self.hop_length = int(fs * hop_duration)  # Convert to samples
        self.fs = fs
        self.n_fft = n_fft
        self.stft_hop = stft_hop
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.dynamic_range = dynamic_range
        self.scale_output = scale_output
        self.audio_paths = audio_paths

    def __len__(self):
        """This loader processes one audio file at a time."""
        return len(self.audio_paths)

    def __getitem__(self, idx) -> np.ndarray:

        # Initialize a worker-specific melspec instance if not already present.
        if not hasattr(self, "mel_spec"):
            self.mel_spec = audio_processing.Melspec_layer(
                segment_duration=self.segment_duration,
                fs=self.fs,
                n_fft=self.n_fft,
                stft_hop=self.stft_hop,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max,
                dynamic_range=self.dynamic_range,
                scale=self.scale_output,
            )

        audio_path = self.audio_paths[idx]

        try:
            audio = es.MonoLoader(
                filename=str(audio_path), sampleRate=self.fs, resampleQuality=0
            )().reshape(-1)
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}: {e}") from e

        if len(audio) < self.segment_length:
            return None, None, audio_path

        # Segment the audio into fixed-length segments
        X_batch, _ = audio_processing.segment_audio(
            audio,
            L=self.segment_length,
            H=self.hop_length,
            discard_remainder=True,  # We discard the remainder
        )

        # Compute mel spectrograms
        X_batch_mel = self.mel_spec.compute_batch(X_batch)
        # Fix the dimensions and types
        X_batch_mel = np.expand_dims(X_batch_mel, 3).astype(np.float32)

        return X_batch, X_batch_mel, audio_path

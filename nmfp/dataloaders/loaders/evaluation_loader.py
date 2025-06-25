import os

import numpy as np

from tensorflow.keras.utils import Sequence

from nmfp import audio_processing


class EvaluationLoader(Sequence):
    """Dataloader object for loading audio clips, and segmenting.
    Augmentation is not possible with this loader.
    """

    def __init__(
        self,
        track_paths: list,
        segment_duration=1,
        hop_duration=0.5,
        fs: float = 8000,
        n_fft=1024,
        stft_hop=256,
        n_mels=256,
        f_min=300,
        f_max=4000,
        dynamic_range=80,
        scale_output=True,
        bsz=120,
    ):
        """
        Parameters
        ----------
            track_paths : list(str),
                Track .wav paths as a list.
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
            bsz : (int), optional
                Generation batch size in segments. The default is 120.
        """

        super(EvaluationLoader, self).__init__()

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

        # Save Input parameters
        self.bsz = bsz

        # Create segment information for each track. This is needed to
        # determine the track boundaries and the amount of space needed.
        # We remove the remainder segments because they are mostly zeros
        self.track_seg_dict = audio_processing.get_track_segment_dict(
            file_paths=track_paths,
            fs=self.fs,
            segment_duration=self.segment_duration,
            hop_duration=self.hop_duration,
            discard_remainder=True,
            skip_short=True,
        )
        print(f"{len(self.track_seg_dict):,} audio files are used.")
        # Create a list of track-segment pairs. We convert it to a list so that
        # each segment can be used during fp-generation.
        self.track_seg_list = [
            [file_path, *seg]
            for file_path, segments in self.track_seg_dict.items()
            for seg in segments
        ]
        self.n_samples = len(self.track_seg_list)
        self.indexes = np.arange(self.n_samples)

    def __len__(self):
        """Returns the number of batches."""

        # Ceil is used to make sure that all the samples are used
        return int(np.ceil(self.n_samples / self.bsz))

    def __getitem__(self, idx) -> np.ndarray:
        """Loads  a batch of segments. A batch may contain segments from different
        tracks. The segments are loaded in a contiguous manner. Therefore, the
        segments of a single track are not necessarily in the same batch.

        Parameters:
        ----------
            idx (int):
                Batch index

        Returns:
        --------
            X_batch (ndarray):
                audio samples (bsz, T)
            X_batch_mel (ndarray):
                power mel-spectrogram of samples (bsz, n_mels, T, 1)

        """

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

        # Get the segments for this batch
        X_batch = []
        for i in self.indexes[idx * self.bsz : (idx + 1) * self.bsz]:
            # Get the information of the segment
            file_path, seg_idx, _, _ = self.track_seg_list[i]

            # Determine the anchor start time
            start_sec = seg_idx * self.hop_duration

            # Load the anchor segment and append it to the batch
            xs = audio_processing.load_wav(
                file_path,
                seg_start_sec=start_sec,
                seg_dur_sec=self.segment_duration,
                fs=self.fs,
                pad_if_short=False,  # should not happen since we discard the remainder
            )
            X_batch.append(xs.reshape((1, -1)))
        # Create the batch of audio
        X_batch = np.concatenate(X_batch, axis=0)

        # Compute mel spectrograms
        X_batch_mel = self.mel_spec.compute_batch(X_batch)
        # Fix the dimensions and types
        X_batch_mel = np.expand_dims(X_batch_mel, 3).astype(np.float32)

        return X_batch, X_batch_mel

    def get_track_information(self):
        """Save the track boundaries and paths. In order to use batch processing,
        we store the database track's segments concatenated. Using this method,
        the boundaries of each database track can be determined."""

        track_boundaries = [[0]]
        track_paths = []
        for i, (track_path, segments) in enumerate(self.track_seg_dict.items()):
            # End index, exclusive
            track_boundaries[-1].append(track_boundaries[-1][0] + len(segments))
            if i < len(self.track_seg_dict) - 1:
                # Start index, inclusive
                track_boundaries.append([track_boundaries[-1][1]])
            track_paths.append(os.path.abspath(track_path))
        track_boundaries = np.vstack(track_boundaries)

        assert (
            track_boundaries[-1, -1] == self.n_samples
        ), "Last boundary does not match the number of samples"
        assert len(track_boundaries) == len(
            self.track_seg_dict
        ), "Something went wrong with the track boundaries."
        assert np.all(
            track_boundaries[:, 0] < track_boundaries[:, 1]
        ), "Some tracks have no segments."
        assert np.all(
            track_boundaries[1:, 0] == track_boundaries[:-1, 1]
        ), "Something went wrong with the track boundaries."

        return track_paths, track_boundaries

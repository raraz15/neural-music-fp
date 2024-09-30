import os
import numpy as np

from tensorflow.keras.utils import Sequence

from nmfp.dataloaders import DataLoader
from nmfp import audio_processing


class GenerationLoader(DataLoader):
    """Dataloader object for loading audio segments. Stores the segment
    inforation of all tracks in a contiguous manner and loads a batch of
    segments at each iteration, which may contain segments from different
    tracks. Augmentation is not possible with this loader. It is either
    used to load augmented query chunks or clean database tracks.
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
                Generation batch size. The default is 120.
        """

        super().__init__(
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            fs=fs,
            n_fft=n_fft,
            stft_hop=stft_hop,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            scale_output=scale_output,
        )

        # Save Input parameters
        self.bsz = bsz

        # Create segment information for each track
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

    def __getitem__(self, idx):
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
                pad_if_short=False,  # should not happen since we discard remainder
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


class GenerationLoaderAugment(Sequence):
    """Dataloader object for augmenting audio chunks. Expects that each audio chunk
    in chunk_paths can fit to the memory. Each batch will contain all the
    segments of a single audio chunk. Augmentation is possible for these segments.
    The augmentation parameters are saved in memory and can be used to write the
    augmented chunks to disk.
    """

    def __init__(
        self,
        chunk_paths: list,
        fs: float = 8000,
        bg_aug_parameters=[False],
        room_ir_aug_parameters=[False],
        mic_ir_aug_parameters=[False],
        shuffle_aug=False,
    ):
        """
        Parameters
        ----------
            chunk_paths : list(str),
                Audio chunk .wav paths as a list.
            fs : (float), optional
                Sampling rate. The default is 8000.
            bg_aug_parameters list([(bool), list(str), (int, int)]):
                [True, BG_FILEPATHS, (MIN_SNR, MAX_SNR), (MIN_AMP, MAX_AMP)].
                The default is [False].
            room_ir_aug_parameters list([(bool), list(str)]):
                [True, IR_FILEPATHS]. The default is [False].
            mic_ir_aug_parameters list([(bool), list(str)]):
                [True, IR_FILEPATHS]. The default is [False].
            shuffle_aug (bool), optional
                Shuffle the augmentation parameters. The default is False.
        """

        # Save the Input parameters
        self.chunk_paths = chunk_paths
        self.fs = fs

        # Save augmentation parameters, read the files, and store them in memory
        self.load_and_store_bg_samples(bg_aug_parameters, shuffle_aug)
        self.load_and_store_room_ir_samples(room_ir_aug_parameters, shuffle_aug)
        self.load_and_store_mic_ir_samples(mic_ir_aug_parameters, shuffle_aug)

        # Store which augmentations are used for each chunk
        self.augmentation_mappings = {}

    def __len__(self):
        """Returns the number of batches. One batch is defined as the set of
        segments  a single chunk. Therefore each batch contains segments
        from a single track."""

        return len(self.chunk_paths)

    def __getitem__(self, idx):
        """Loads a single chunk of audio. If bg_mix and ir_mix are False, we simply
        segment the chunk and compute the power mel-spectrogram for each segment.
        If both bg_mix and ir_mix are True, we first augment the chunk with background
        noise and convolve it with a room Impulse Response (IR) in cascade. After the
        augmentation, the augmented chunk is segmented and the power mel-spectrogram
        is computed for each segment.

        Parameters:
        ----------
            idx (int):
                Chunk index

        Returns:
        --------
            Xa (ndarray):
                audio samples of the chunk (T,)
            Xp (ndarray or None):
                augmented chunk samples (n_segments, T)
        """

        # Load the chunk. In real-life segments or chunks are not normalized
        # A chunk should not require padding
        Xa = audio_processing.load_wav(
            self.chunk_paths[idx],
            fs=self.fs,
            normalize=False,
            pad_if_short=False,
        )
        Xa = Xa.reshape((1, -1))  # (1,T)

        # If no augmentation is applied, we simply return the chunk
        if not (self.bg_aug or self.room_ir_aug or self.mic_ir_aug):
            print("No augmentation is applied.")
            Xa = Xa.reshape(-1)
            return Xa, None

        else:
            # Save the augmentation mapping for future use
            self.augmentation_mappings[idx] = {"music_path": self.chunk_paths[idx]}
            # Copy the chunk to avoid changing the original chunk
            Xp = Xa.copy()

            """First, we apply the background noise augmentation."""
            if self.bg_aug:
                # Read a single bg noise sample from memory
                bg_path = self.bg_paths[idx % self.n_bg_files]
                bg = self.read_bg(bg_path, Xa.shape[1])
                # Mix the chunk with the bg noise with a random SNR from the range
                Xp, SNR = audio_processing.bg_mix_batch(
                    Xp, bg, snr_range=self.bg_snr_range
                )
                # Save the augmentation mapping for future use
                self.augmentation_mappings[idx]["bg_path"] = bg_path
                self.augmentation_mappings[idx]["bg_snr"] = SNR[0]

            """Then we apply the Room IR"""
            if self.room_ir_aug:
                # Read a single room ir sample from memory
                room_ir_path = self.room_ir_paths[idx % self.n_room_ir_clips]
                room_ir = self.room_ir_clips[room_ir_path]
                # We apply random gain before the Room IR augmentation if specified
                if self.rir_random_gain_range != [1, 1]:
                    Xp, rand_gain = audio_processing.apply_random_gain_batch(
                        Xp, self.rir_random_gain_range
                    )
                    # Save the augmentation mapping for future use
                    self.augmentation_mappings[idx]["pre_rir_mix_gain"] = rand_gain[0]
                # Apply Room IR
                Xp = audio_processing.convolve_with_IR_batch(Xp, [room_ir])
                # Save the augmentation mapping for future use
                self.augmentation_mappings[idx]["room_ir_path"] = room_ir_path
                self.augmentation_mappings[idx]["room_ir_duration"] = (
                    len(room_ir) / self.fs
                )

            """Finally, we apply the Microphone IR"""
            if self.mic_ir_aug:
                # Read a single mic ir sample from memory
                mic_ir_path = self.mic_ir_paths[idx % self.n_mic_ir_clips]
                mic_ir = self.mic_ir_clips[mic_ir_path]
                # We apply random gain before the Microphone IR augmentation if specified
                if self.mir_random_gain_range != [1, 1]:
                    Xp, rand_gain = audio_processing.apply_random_gain_batch(
                        Xp, self.mir_random_gain_range
                    )
                    # Save the augmentation mapping for future use
                    self.augmentation_mappings[idx]["pre_mir_mix_gain"] = rand_gain[0]
                # Apply Microphone IR
                Xp = audio_processing.convolve_with_IR_batch(Xp, [mic_ir])
                # Save the augmentation mapping for future use
                self.augmentation_mappings[idx]["mic_ir_path"] = mic_ir_path
                self.augmentation_mappings[idx]["mic_ir_duration"] = (
                    len(mic_ir) / self.fs
                )

            # Reshape the chunks back to (T,)
            Xa = Xa.reshape(-1)
            Xp = Xp.reshape(-1)

            return Xa, Xp

    def read_bg(self, file_path, chunk_length):
        """Read a single background noise clip for the given file_path from the memory.
        Each sample is repeated or cut to chunk_length.

        Parameters:
        ----------
            file_path (str):
                Background noise file path.
            chunk_length (int):
                Length of the chunk in samples.

        Returns:
        --------
            X_bg (ndarray):
                (1,chunk_length)
        """

        # Read the bg noise sample from memory
        X_bg = self.bg_clips[file_path]

        # If the bg noise is shorter than the chunk length, we repeat it
        N = len(X_bg)
        if N < chunk_length:
            n_repeats = int(np.ceil(chunk_length / N))
            X_bg = np.tile(X_bg, n_repeats)
            X_bg = X_bg[:chunk_length]
        elif N > chunk_length:  # Otherwise we cut a random part
            start = np.random.randint(0, N - chunk_length)
            X_bg = X_bg[start : start + chunk_length]

        return X_bg.reshape((1, -1))  # (1, T)

    def load_and_store_bg_samples(self, bg_mix_parameter, shuffle_aug):
        """Loads background noise samples in memory and saves the augmentation
        parameters.

        Parameters:
        ----------
            bg_aug_parameters [(bool), list(str), (float, float)]:
                [True, BG_FILEPATHS, (MIN_SNR, MAX_SNR)].
            shuffle_aug (bool):
                Shuffle the augmentation file paths.
        """

        self.bg_aug = bg_mix_parameter[0]
        if self.bg_aug:
            # Record parameters
            self.bg_paths = bg_mix_parameter[1]
            self.bg_snr_range = bg_mix_parameter[2]

            # Shuffle the bg_file_paths
            if shuffle_aug:
                np.random.shuffle(self.bg_paths)

            # Load all bg clips in full duration
            print("Loading Background Noise clips in memory...")
            self.bg_clips = {}
            for file_path in self.bg_paths:
                self.bg_clips[file_path] = audio_processing.load_wav(
                    file_path, fs=self.fs
                )
            self.n_bg_files = len(self.bg_clips)
            print(f"{self.n_bg_files:,} BG clips are used.")
        else:
            self.bg_paths = []
            self.bg_snr_range = [None, None]

    def load_and_store_room_ir_samples(self, room_ir_aug_parameters, shuffle_aug):
        """Load Room Impulse Response samples in memory. Room IRs are
        loaded in full duration and NOT truncated.

        Parameters:
        ----------
            room_ir_aug_parameters [(bool), list(str), (float, float)]:
                [True, IR_FILEPATHS, (MIN_AMP, MAX_AMP)].
            shuffle_aug (bool):
                Shuffle the augmentation file paths.
        """

        self.room_ir_aug = room_ir_aug_parameters[0]
        if self.room_ir_aug:
            # Save the file paths
            self.room_ir_paths = room_ir_aug_parameters[1]
            self.rir_random_gain_range = room_ir_aug_parameters[2]

            # Shuffle the ir_file_paths
            if shuffle_aug:
                np.random.shuffle(self.room_ir_paths)

            # Load all Room IR clips in full duration
            print("Loading Room Impulse Response samples in memory...")
            self.room_ir_clips = {}
            for file_path in self.room_ir_paths:
                self.room_ir_clips[file_path] = audio_processing.load_wav(
                    file_path,
                    fs=self.fs,
                )
            self.n_room_ir_clips = len(self.room_ir_clips)
            print(f"{self.n_room_ir_clips:,} Room IR clips are used.")
        else:
            self.room_ir_paths = []
            self.rir_random_gain_range = [None, None]

    def load_and_store_mic_ir_samples(self, mic_ir_aug_parameters, shuffle_aug):
        """Load Microphone Impulse Response samples in memory. These segments are
        truncated to self.total_context_duration. Since the audio segment
        are of this duration, using the full IR has no effect on the output.
        Therefore we only keep the relevant part of the IR in memory.

        Parameters:
        ----------
            mic_ir_aug_parameters [(bool), list(str), (float, float)]:
                [True, IR_FILEPATHS, (MIN_AMP, MAX_AMP)].
            shuffle_aug (bool):
                Shuffle the augmentation file paths.
        """

        self.mic_ir_aug = mic_ir_aug_parameters[0]
        if self.mic_ir_aug:
            # Save the file paths
            self.mic_ir_paths = mic_ir_aug_parameters[1]
            self.mir_random_gain_range = mic_ir_aug_parameters[2]

            # Shuffle the ir_file_paths
            if shuffle_aug:
                np.random.shuffle(self.mic_ir_paths)

            # Load all Microphone IR clips in full duration
            print("Loading Microphone Impulse Response samples in memory...")
            self.mic_ir_clips = {}
            for file_path in self.mic_ir_paths:
                self.mic_ir_clips[file_path] = audio_processing.load_wav(
                    file_path,
                    fs=self.fs,
                )
            self.n_mic_ir_clips = len(self.mic_ir_clips)
            print(f"{self.n_mic_ir_clips:,} Microphone IR clips are used.")
        else:
            self.mic_ir_paths = []
            self.mir_random_gain_range = [None, None]

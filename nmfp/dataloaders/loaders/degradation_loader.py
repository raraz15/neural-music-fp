import numpy as np

from tensorflow.keras.utils import Sequence

from nmfp import audio_processing


class DegradationLoader(Sequence):
    """Dataloader object for degrading audio. Expects that each track in track_paths
    can fit to the memory. Each batch will contain all the segments of a single track. 
    Degradation is possible for these segments. The degradation parameters are saved 
    in memory and can be used to write the degraded tracks to disk."""

    def __init__(
        self,
        track_paths: list,
        fs: float = 8000,
        bg_aug_parameters=[False],
        room_ir_aug_parameters=[False],
        mic_ir_aug_parameters=[False],
        shuffle_aug=False,
    ):
        """
        Parameters
        ----------
            track_paths : list(str),
                Full track .wav paths as a list.
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
                Shuffle the degradation parameters. The default is False.
        """

        super(DegradationLoader, self).__init__()

        # Save the input parameters
        self.track_paths = track_paths
        self.fs = fs

        # Save degradation parameters, read the files, and store them in memory
        self.load_and_store_bg_samples(bg_aug_parameters, shuffle_aug)
        self.load_and_store_room_ir_samples(room_ir_aug_parameters, shuffle_aug)
        self.load_and_store_mic_ir_samples(mic_ir_aug_parameters, shuffle_aug)

        # Store degradation history
        self.degradation_mappings = {}

    def __len__(self):
        """Returns the number of batches. One batch is defined as the set of
        segments of a single track."""

        return len(self.track_paths)

    def __getitem__(self, idx):
        """Loads a single track. If bg_mix and ir_mix are False, we simply segment the 
        track and compute the power mel-spectrogram for each segment. If both bg_mix and 
        ir_mix are True, we first degrade the track with background noise and convolve 
        it with a room Impulse Response (IR) in cascade. After the degradation, the 
        degraded track is segmented and the power mel-spectrogram is computed for each 
        segment.

        Parameters:
        ----------
            idx (int):
                Track index

        Returns:
        --------
            Xa (ndarray):
                audio (T,)
            Xp (ndarray or None):
                degradeed audio (n_segments, T)
        """

        # Load the track. In real-life segments or tracks are not normalized
        # A track should not require padding
        Xa = audio_processing.load_wav(
            self.track_paths[idx],
            fs=self.fs,
            normalize=False,
            pad_if_short=False,
        )
        Xa = Xa.reshape((1, -1))  # (1,T)

        # If no degradation is applied, we simply return the track
        if not (self.bg_aug or self.room_ir_aug or self.mic_ir_aug):
            print("No degradation is applied.")
            Xa = Xa.reshape(-1)
            return Xa, None

        else:
            # Save the degradation mapping for future use
            self.degradation_mappings[idx] = {"music_path": self.track_paths[idx]}
            # Copy the track to avoid changing the original
            Xp = Xa.copy()

            """First, we apply the background noise degradation."""
            if self.bg_aug:
                # Read a single bg noise sample from memory
                bg_path = self.bg_paths[idx % self.n_bg_files]
                bg = self.read_bg(bg_path, Xa.shape[1])
                # Mix the track with the bg noise with a random SNR from the range
                Xp, SNR = audio_processing.bg_mix_batch(
                    Xp, bg, snr_range=self.bg_snr_range
                )
                # Save the degradation mapping for future use
                self.degradation_mappings[idx]["bg_path"] = bg_path
                self.degradation_mappings[idx]["bg_snr"] = SNR[0]

            """Then we apply the Room IR"""
            if self.room_ir_aug:
                # Read a single room ir sample from memory
                room_ir_path = self.room_ir_paths[idx % self.n_room_ir_clips]
                room_ir = self.room_ir_clips[room_ir_path]
                # We apply random gain before the Room IR degradation if specified
                if self.rir_random_gain_range != [1, 1]:
                    Xp, rand_gain = audio_processing.apply_random_gain_batch(
                        Xp, self.rir_random_gain_range
                    )
                    # Save the degradation mapping for future use
                    self.degradation_mappings[idx]["pre_rir_mix_gain"] = rand_gain[0]
                # Apply Room IR
                Xp = audio_processing.convolve_with_IR_batch(Xp, [room_ir])
                # Save the degradation mapping for future use
                self.degradation_mappings[idx]["room_ir_path"] = room_ir_path
                self.degradation_mappings[idx]["room_ir_duration"] = (
                    len(room_ir) / self.fs
                )

            """Finally, we apply the Microphone IR"""
            if self.mic_ir_aug:
                # Read a single mic ir sample from memory
                mic_ir_path = self.mic_ir_paths[idx % self.n_mic_ir_clips]
                mic_ir = self.mic_ir_clips[mic_ir_path]
                # We apply random gain before the Microphone IR degradation if specified
                if self.mir_random_gain_range != [1, 1]:
                    Xp, rand_gain = audio_processing.apply_random_gain_batch(
                        Xp, self.mir_random_gain_range
                    )
                    # Save the degradation mapping for future use
                    self.degradation_mappings[idx]["pre_mir_mix_gain"] = rand_gain[0]
                # Apply Microphone IR
                Xp = audio_processing.convolve_with_IR_batch(Xp, [mic_ir])
                # Save the degradation mapping for future use
                self.degradation_mappings[idx]["mic_ir_path"] = mic_ir_path
                self.degradation_mappings[idx]["mic_ir_duration"] = (
                    len(mic_ir) / self.fs
                )

            # Reshape the tracks back to (T,)
            Xa = Xa.reshape(-1)
            Xp = Xp.reshape(-1)

            return Xa, Xp

    def read_bg(self, file_path, track_length):
        """Read a single background noise clip for the given file_path from the memory.
        Each sample is repeated or cut to track_length.

        Parameters:
        ----------
            file_path (str):
                Background noise file path.
            track_length (int):
                Length of the track in samples.

        Returns:
        --------
            X_bg (ndarray):
                (1,track_length)
        """

        # Read the bg noise sample from memory
        X_bg = self.bg_clips[file_path]

        # If the bg noise is shorter than the track length, we repeat it
        N = len(X_bg)
        if N < track_length:
            n_repeats = int(np.ceil(track_length / N))
            X_bg = np.tile(X_bg, n_repeats)
            X_bg = X_bg[:track_length]
        elif N > track_length:  # Otherwise we cut a random part
            start = np.random.randint(0, N - track_length)
            X_bg = X_bg[start : start + track_length]

        return X_bg.reshape((1, -1))  # (1, T)

    def load_and_store_bg_samples(self, bg_mix_parameter, shuffle_aug):
        """Loads background noise samples in memory and saves the degradation
        parameters.

        Parameters:
        ----------
            bg_aug_parameters [(bool), list(str), (float, float)]:
                [True, BG_FILEPATHS, (MIN_SNR, MAX_SNR)].
            shuffle_aug (bool):
                Shuffle the degradation file paths.
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
                Shuffle the degradation file paths.
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
                Shuffle the degradation file paths.
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

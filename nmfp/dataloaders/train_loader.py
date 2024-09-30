import numpy as np

from nmfp import audio_processing
from nmfp.dataloaders import DataLoader


class TrainLoader(DataLoader):
    """DataLoader for training. At each iteration it loads a batch of anchor
    (original) and positive (replica) samples of audio. Each anchor sample in
    a batch belongs to a different track and the positive samples are created
    by randomly offseting the anchor samples within the bounds of the segment
    and they are augmented with background noise and impulse responses. The power
    mel-spectrograms of the anchor and positive samples are computed and returned
    as well. We follow a strategy that allows us to all the segments of a track
    once per epoch. We also use a past context for more realistic degradations.
    """

    def __init__(
        self,
        chunk_paths: list,
        segment_duration=1,
        hop_duration=0.5,
        past_context_duration=0.0,
        fs: float = 8000,
        n_fft=1024,
        stft_hop=256,
        n_mels=256,
        f_min=300,
        f_max=4000,
        scale_output=True,
        segments_per_track=59,
        bsz=120,
        shuffle=False,
        max_offset_dur_anch=0.25,  # TODO delete?
        max_offset_dur_pos=0.25,
        bg_deg_parameters=[False],
        room_ir_deg_parameters=[False],
        mic_ir_deg_parameters=[False],
    ):
        """
        Parameters:
        -----------
            chunk_paths list(str):
                list of audio chunk paths in the dataset. The chunks
                are not segmented, we segment them here. They should be
                approximately equal in duration.
            segment_duration float:
                duration of the segments in seconds.
            hop_duration float:
                hop duration in seconds.
            past_context_duration float:
                duration of the past context in seconds for more realistic
                degradations during training.
            fs : (int), optional
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
            segments_per_track : (int), optional
                Number of segments per track. The default is 59. Tracks
                with less than segments_per_track are discarded.
            bsz : (int), optional
                Batch size. The default is 120.
            shuffle : (bool), optional
                Shuffle the order of segments and chunks before training.
                The default is False.
            max_offset_dur_anch : (float), optional
                Maximum offset duration in seconds for the anchor segment.
                The default is 0.25.
            max_offset_dur_pos : (float), optional
                Maximum offset duration in seconds for the positive segment.
                The default is 0.25.
            bg_deg_parameters list([(bool), list(str), (int, int)]):
                [True, BG_FILEPATHS, (MIN_SNR, MAX_SNR), (MIN_AMP, MAX_AMP)].
                The default is [False].
            room_ir_deg_parameters list([(bool), list(str)]):
                [True, IR_FILEPATHS]. The default is [False].
            mic_ir_deg_parameters list([(bool), list(str)]):
                [True, IR_FILEPATHS]. The default is [False].
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

        # Check parameters
        assert bsz % 2 == 0, "bsz should be even"
        assert segments_per_track > 0, "segments_per_track should be > 0"
        assert past_context_duration >= 0, "past_context_duration should be >= 0"
        assert max_offset_dur_pos >= 0, "max_offset_dur_pos should be >= 0"
        assert max_offset_dur_anch >= 0, "max_offset_dur_anch should be >= 0"
        assert (
            max_offset_dur_pos < hop_duration
        ), "max_offset_dur_pos should be < hop_duration"
        assert (
            max_offset_dur_anch < hop_duration
        ), "max_offset_dur_anch should be < hop_duration"

        # Save the duration parameters
        self.chunk_paths = chunk_paths
        self.hop_duration = hop_duration
        self.past_context_duration = past_context_duration
        self.total_context_duration = self.segment_duration + self.past_context_duration
        self.max_offset_dur_anch = max_offset_dur_anch
        self.max_offset_dur_pos = max_offset_dur_pos
        # Convert to samples
        self.hop_length = int(fs * self.hop_duration)
        self.past_context_length = int(fs * self.past_context_duration)
        self.total_context_length = self.segment_length + self.past_context_length
        self.max_offset_len_anch = int(max_offset_dur_anch * fs)  # Convert to samples
        self.max_offset_len_pos = int(max_offset_dur_pos * fs)  # Convert to samples

        # Other parameters
        self.segments_per_track = segments_per_track
        self.shuffle = shuffle
        self.bsz = bsz
        # Half of the batch is positive samples
        self.n_pos = bsz // 2
        # The rest are anchors
        self.n_anchor = bsz - self.n_pos
        assert self.n_anchor == self.n_pos, "n_anchor should be equal to n_pos"

        # Create segment information for each track. We keep the remainder
        # segments for training since in FMA_mdium the remainder is almost a
        # full segment. We segment the chunk with segment_duration to use more
        # music but consider the past_context_duration for the degradation
        self.track_seg_dict = audio_processing.get_track_segment_dict(
            chunk_paths,
            fs=self.fs,
            segment_duration=self.segment_duration,
            hop_duration=self.hop_duration,
            discard_remainder=False,
        )

        # Keep only segments_per_track segments for each track
        keys_to_remove = []
        for k, v in self.track_seg_dict.items():
            if len(v) < segments_per_track:
                print(f"{k} has {len(v)} segments only. Removing it from the dataset")
                keys_to_remove.append(k)
            else:
                self.track_seg_dict[k] = v[:segments_per_track]
        # Remove the tracks with less than segments_per_track
        if len(keys_to_remove) > 0:
            for k in keys_to_remove:
                del self.track_seg_dict[k]
            print(f"{len(self.track_seg_dict):,} tracks remaining.")

        # Remove the tracks that do not fill the last batch. Each batch contains
        # a single segment from n_anchor different tracks.
        self.n_tracks = int((len(self.track_seg_dict) // self.n_anchor) * self.n_anchor)
        self.track_seg_dict = {
            k: v
            for i, (k, v) in enumerate(self.track_seg_dict.items())
            if i < self.n_tracks
        }
        self.track_file_paths = list(self.track_seg_dict.keys())
        print(
            f"{len(self.track_file_paths):,} tracks are used for mini-batch processing."
        )
        self.n_samples = sum([len(v) for v in self.track_seg_dict.values()])
        assert self.n_samples == self.n_tracks * self.segments_per_track, (
            f"n_samples {self.n_samples} should be equal to "
            f"n_tracks {self.n_tracks} * segments_per_track {self.segments_per_track}"
        )
        print(f"{self.n_samples:,} number of segments used per epoch.")

        # Save degradation parameters, read the files, and store them in memory
        self.load_and_store_bg_samples(bg_deg_parameters)
        self.load_and_store_room_ir_samples(room_ir_deg_parameters)
        self.load_and_store_mic_ir_samples(mic_ir_deg_parameters)
        # If no degradation is applied, we simply return the positive sample
        if not (self.bg_aug or self.room_ir_aug or self.mic_ir_aug):
            print("No degradations will be applied to the positive sample.")
            assert (
                self.max_offset_len_pos > 0 and self.max_offset_len_anch > 0
            ), "If no degradation is applied, max_offset_len_pos and max_offset_len_anch should be > 0."

        # Shuffle if specified
        if self.shuffle:
            self.shuffle_events()
            self.shuffle_segments()

    def __len__(self):
        """Returns the number of batches per epoch. An epoch is defined as
        when all the segments of each track are seen once."""

        return int(self.n_samples // self.n_anchor)

    def __getitem__(self, idx):
        """Get a batch of anchor (original) and positive (replica) samples of audio
        and their power mel-spectrograms. During training we follow a strategy that
        allows us to use all segments of each track, which is predifened during
        initialization.

        Parameters:
        ----------
            idx (int):
                Batch index

        Returns:
        --------
            Xa_batch (ndarray):
                anchor audio samples (self.n_anchor, T)
            Xp_batch (ndarray):
                positive audio samples (self.n_pos, T)
            Xa_batch_mel (ndarray):
                power mel-spectrogram of anchor samples (self.n_anchor, n_mels, T, 1)
            Xp_batch_mel (ndarray):
                power-mel spectrogram of positive samples (self.n_pos, n_mels, T, 1)
        """

        # Indices of tracks to use in this batch
        i0, i1 = idx * self.n_anchor, (idx + 1) * self.n_anchor
        # Get their filenames, each file will be used self.segments_per_track
        # times during an epoch
        file_paths = [self.track_file_paths[i % self.n_tracks] for i in range(i0, i1)]

        # Load anchor and positive audio samples for each filename
        Xa_batch, Xp_batch = self.batch_load_track_segments(file_paths, idx)

        # If all degradations are specified, we apply them in sequence to the chunk
        """First, we apply the background noise degradation."""
        if self.bg_aug:
            # Prepare BG for positive samples
            bg_fnames = [self.bg_file_paths[i % self.n_bg_clips] for i in range(i0, i1)]
            bg_batch = self.batch_read_bg(bg_fnames, idx)
            # Mix
            Xp_batch, _ = audio_processing.bg_mix_batch(
                Xp_batch, bg_batch, snr_range=self.bg_snr_range
            )

        """ Then we apply the Room IR"""
        if self.room_ir_aug:
            # Prepare Room IR for positive samples
            room_ir_paths = [
                self.room_ir_paths[i % self.n_room_ir_clips] for i in range(i0, i1)
            ]
            room_ir_batch = [
                self.room_ir_clips[file_path] for file_path in room_ir_paths
            ]

            # To simulate distance we can optionally apply random gain before the Room IR
            if self.rir_random_gain_range != [1, 1]:
                Xp_batch, _ = audio_processing.apply_random_gain_batch(
                    Xp_batch, self.rir_random_gain_range
                )
            # Apply Room IR
            Xp_batch = audio_processing.convolve_with_IR_batch(Xp_batch, room_ir_batch)

            # Remove the past context from the positive samples
            if self.past_context_length > 0:
                Xp_batch = Xp_batch[:, self.past_context_length :]

        """ Finally, we apply the Microphone IR"""
        if self.mic_ir_aug:
            # Prepare Microphone IR for positive samples
            mic_ir_paths = [
                self.mic_ir_paths[i % self.n_mic_ir_clips] for i in range(i0, i1)
            ]
            mic_ir_batch = [self.mic_ir_clips[file_path] for file_path in mic_ir_paths]

            # To simulate distance we can optionally apply random gain before the Microphone IR
            if self.mir_random_gain_range != [1, 1]:
                Xp_batch, _ = audio_processing.apply_random_gain_batch(
                    Xp_batch, self.mir_random_gain_range
                )
            # Apply Microphone IR
            Xp_batch = audio_processing.convolve_with_IR_batch(Xp_batch, mic_ir_batch)
            # We do not use the past context here to simulate real life conditions

        # Compute mel spectrograms
        Xa_batch_mel = self.mel_spec.compute_batch(Xa_batch).astype(np.float32)
        Xp_batch_mel = self.mel_spec.compute_batch(Xp_batch).astype(np.float32)

        # Fix the dimensions
        Xa_batch_mel = np.expand_dims(Xa_batch_mel, 3)
        Xp_batch_mel = np.expand_dims(Xp_batch_mel, 3)

        return Xa_batch, Xp_batch, Xa_batch_mel, Xp_batch_mel

    def batch_load_track_segments(self, file_paths, idx):
        """Load a segment conditioned on idx from the current track. Since
        we shuffle the segments of each track, we can use idx to get a
        different segment from each track. We also load 1 positive sample for each anchor.
        The anchor is randomly offsetted within the bounds of the segment. The
        positive sample is randomly offsetted within the bounds of the segment based on
        the offset of the anchor. We also use a past context for more realistic
        degradations during training.

        Parameters:
        ----------
            file_paths list(int):
                list of the audio paths of the current batch.

        Returns:
        --------
            Xa_batch (ndarray):
                (n_anchor, T)
            Xp_batch (ndarray):
                (n_anchor, T)
        """

        # If segments_per_track is even, each epoch half of the segments will be
        # seen twice and the other half zero. We want to see all segments once,
        # so we make sure that the second half of the segments are seen too.
        if self.segments_per_track % 2 == 0 and idx >= self.__len__() / 2:
            idx = (idx + 1) % self.segments_per_track
        else:
            # If segments_per_track is odd, all segments will be seen once each epoch
            idx = idx % self.segments_per_track

        Xa_batch, Xp_batch = [], []
        for file_path in file_paths:
            # Get the segment information of the random_idx segment of the track
            seg_idx, offset_min, offset_max = self.track_seg_dict[file_path][idx]

            # Determine the anchor start time
            seg_start_sec = seg_idx * self.hop_duration

            # If specified, sample a random offset for the anchor segment.
            if self.max_offset_len_anch > 0:
                # Sample a random offset sample
                anchor_offset_min = np.max([offset_min, -self.max_offset_len_anch])
                anchor_offset_max = np.min([offset_max, self.max_offset_len_anch])
                _anchor_offset_sample = np.random.randint(
                    low=anchor_offset_min, high=anchor_offset_max
                )
                _anchor_offset_sec = _anchor_offset_sample / self.fs
            else:
                _anchor_offset_sample, _anchor_offset_sec = 0, 0

            # Apply the offset to the anchor start time
            anchor_start_sec = seg_start_sec + _anchor_offset_sec
            assert anchor_start_sec >= 0, "Anchor start point is out of bounds."
            # I will not check the upper bound because it can create problems
            # when segments_per_track is set to shorter than the actual number

            # Load the anchor sample from the chunk
            # We pad short segments May happen but with FMA_medium
            # the remainder is almost a full segment
            Xa = audio_processing.load_wav(
                file_path,
                seg_start_sec=anchor_start_sec,
                seg_dur_sec=self.segment_duration,
                fs=self.fs,
                pad_if_short=True,
            )
            Xa_batch.append(Xa.reshape((1, -1)))

            # Based on the anchor offset time, sample an offset for positive sample Make sure
            # that the offset is within the bounds of the segment.
            if self.max_offset_len_pos > 0:
                pos_offset_min = np.max(
                    [-self.max_offset_len_pos, offset_min - _anchor_offset_sample]
                )
                pos_offset_max = np.min(
                    [+self.max_offset_len_pos, offset_max - _anchor_offset_sample]
                )
                _pos_offset_sample = np.random.randint(
                    low=pos_offset_min, high=pos_offset_max
                )
                _pos_offset_sec = _pos_offset_sample / self.fs
                # Shift the positive sample with respect to the anchor
                pos_start_sec = anchor_start_sec + _pos_offset_sec
                assert pos_start_sec >= 0, "Start point is out of bounds"
            else:
                pos_start_sec = anchor_start_sec

            # If there is enough past context, adjust the start time of the positive sample
            if pos_start_sec > self.past_context_duration:
                # Load the positive sample
                Xp = audio_processing.load_wav(
                    file_path,
                    seg_start_sec=pos_start_sec - self.past_context_duration,
                    seg_dur_sec=self.segment_duration + self.past_context_duration,
                    fs=self.fs,
                    pad_if_short=True,
                )
            else:
                Xp = audio_processing.load_wav(
                    file_path,
                    seg_start_sec=0,
                    seg_dur_sec=self.segment_duration + pos_start_sec,
                    fs=self.fs,
                    pad_if_short=True,
                )
                # Pad the beginning with zeros
                Xp = np.concatenate(
                    [np.zeros((self.total_context_length - len(Xp),)), Xp]
                )
            Xp_batch.append(Xp.reshape((1, -1)))

        # Create a batch
        Xa_batch = np.concatenate(Xa_batch, axis=0)
        Xp_batch = np.concatenate(Xp_batch, axis=0)

        return Xa_batch, Xp_batch

    def batch_read_bg(self, file_paths, index):
        """Read len(file_paths) background samples from the memory. We randomly
        coose a different segment at every iteration conditioned on index.

        Parameters:
        -----------
            file_paths list(str):
                list of background file_paths in the dataset.
            index (int):
                batch index to condition the sample selection.

        Returns:
        --------
            X_bg_batch (ndarray):
                (self.n_pos, T)

        """

        X_bg_batch = []
        for fname in file_paths:
            # Read the complete background noise sample from memory
            X_bg = self.bg_clips[fname]

            # Get the segment information of the file
            bg_segments = self.bg_seg_dict[fname]

            # Choose a different segment every iteration
            seg_idx, _, _ = bg_segments[index % len(bg_segments)]

            # Get the corresponding segment from full clip
            start_idx = seg_idx * self.bg_hop_length
            end_idx = start_idx + self.bg_segment_length
            X_bg = X_bg[start_idx:end_idx]

            # If the background segment is shorter than self.bg_segment_length,
            # repeat. Should not happen with current settings
            N = len(X_bg)
            if N < self.bg_segment_length:
                n_repeats = int(np.ceil(self.bg_segment_length / N))
                X_bg = np.tile(X_bg, n_repeats)
                X_bg = X_bg[: self.bg_segment_length]
            X_bg_batch.append(X_bg)

        # Concatenate the samples
        X_bg_batch = np.stack(X_bg_batch, axis=0)

        return X_bg_batch

    def on_epoch_end(self):
        """Routines to apply at the end of each epoch."""

        # Shuffle all  events if specified
        if self.shuffle:
            self.shuffle_events()
            self.shuffle_segments()

    def shuffle_events(self):
        """Shuffle all events."""

        # Shuffle the order of tracks
        np.random.shuffle(self.track_file_paths)

        # Shuffle the order of degradation types
        if self.bg_aug:
            np.random.shuffle(self.bg_file_paths)
        if self.room_ir_aug:
            np.random.shuffle(self.room_ir_paths)
        if self.mic_ir_aug:
            np.random.shuffle(self.mic_ir_paths)

    def shuffle_segments(self):
        """Shuffle the order of segments of each track and background noise.
        We do not shuffle the order of IRs because we do not segment them."""

        # Shuffle the order of segments of each track
        for file_path in self.track_file_paths:
            np.random.shuffle(self.track_seg_dict[file_path])

        # Shuffle the order of bg segments
        if self.bg_aug:
            for file_path in self.bg_file_paths:
                np.random.shuffle(self.bg_seg_dict[file_path])

    def load_and_store_bg_samples(self, bg_deg_parameters):
        """Load background noise samples in memory and their segmentation
        information.

        Parameters:
        ----------
            bg_deg_parameters [(bool), list(str), (float, float)]:
                [True, BG_FILEPATHS, (MIN_SNR, MAX_SNR)].
        """

        self.bg_aug = bg_deg_parameters[0]
        if self.bg_aug:
            # Record parameters
            self.bg_snr_range = bg_deg_parameters[2]

            # We use past context as well for the bg samples
            self.bg_segment_duration = self.total_context_duration
            self.bg_segment_length = self.total_context_length

            # We use the same hop duration as the original samples
            self.bg_hop_duration = self.hop_duration
            self.bg_hop_length = self.hop_length

            # Get the segment information of the background noise files
            self.bg_seg_dict = audio_processing.get_track_segment_dict(
                bg_deg_parameters[1],
                fs=self.fs,
                segment_duration=self.bg_segment_duration,
                hop_duration=self.bg_hop_duration,
                discard_remainder=False,
                skip_short=False,  # Keep all segments
            )
            self.bg_file_paths = list(self.bg_seg_dict.keys())

            # Load all bg clips in full duration
            print("Loading Background Noise clips in memory...")
            self.bg_clips = {}
            for file_path in self.bg_file_paths:
                self.bg_clips[file_path] = audio_processing.load_wav(
                    file_path, fs=self.fs
                )
            self.n_bg_clips = len(self.bg_clips)
            print(f"{self.n_bg_clips:,} BG clips are used.")

    def load_and_store_room_ir_samples(self, room_ir_deg_parameters):
        """Load Room Impulse Response samples in memory. These segments are
        truncated to self.total_context_duration. Since the audio segments
        are of this duration, using the full IR has no effect on the output.
        Therefore we only keep the relevant part of the IR in memory.

        Parameters:
        ----------
            room_ir_deg_parameters [(bool), list(str), (float, float)]:
                [True, IR_FILEPATHS, (MIN_AMP, MAX_AMP)].
        """

        self.room_ir_aug = room_ir_deg_parameters[0]
        if self.room_ir_aug:
            # Record parameters
            self.rir_random_gain_range = room_ir_deg_parameters[2]

            # Load all Room IR clips
            print("Loading Impulse Response samples in memory...")
            self.room_ir_clips = {}
            for file_path in room_ir_deg_parameters[1]:
                room_ir = audio_processing.load_wav(
                    file_path,
                    fs=self.fs,
                )
                # Truncate IR to total_context_length as we do not need the rest
                self.room_ir_clips[file_path] = room_ir[: self.total_context_length]
            self.n_room_ir_clips = len(self.room_ir_clips)
            self.room_ir_paths = list(self.room_ir_clips.keys())
            print(f"{self.n_room_ir_clips:,} Room IR clips are used.")

    def load_and_store_mic_ir_samples(self, mic_ir_deg_parameters):
        """Load Microphone Impulse Response samples in memory. These segments are
        truncated to self.segment_duration. Since the output segments of the RIR
        convolution are of this duration, using the full IR has no effect on the
        output. Therefore we only keep the relevant part of the IR in memory.

        Parameters:
        ----------
            mic_ir_deg_parameters [(bool), list(str), (float, float)]:
                [True, IR_FILEPATHS, (MIN_AMP, MAX_AMP)].
        """

        self.mic_ir_aug = mic_ir_deg_parameters[0]
        if self.mic_ir_aug:
            # Record parameters
            self.mir_random_gain_range = mic_ir_deg_parameters[2]

            # Load all Microphone IR clips
            print("Loading Microphone Impulse Response samples in memory...")
            self.mic_ir_clips = {}
            for file_path in mic_ir_deg_parameters[1]:
                # Load the IR in full duration first
                mic_ir = audio_processing.load_wav(
                    file_path,
                    fs=self.fs,
                )
                # Truncate IR to segment_length
                self.mic_ir_clips[file_path] = mic_ir[: self.segment_length]
            self.mic_ir_paths = list(self.mic_ir_clips.keys())
            self.n_mic_ir_clips = len(self.mic_ir_clips)
            print(f"{self.n_mic_ir_clips:,} Microphone IR clips are used.")

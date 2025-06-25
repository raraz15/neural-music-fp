""" I wanted to use essentia for computing the Mel-spectrogram instead of what NAFP uses.
In the process I had to find the corresponding parameters for the MelBands and Spectrogram
extraction. I am sure of all the parameters minus 2, whic are indicated with TODOs."""

import numpy as np

import essentia.standard as es


class Melspec_layer:
    """Computes the Mel-spectrogram of an audio segment. The audio segment is padded
    from both sides to center the first window. The Mel-spectrogram is computed using
    Essentia, which only works with 'single' datatype (np.float32). If specified, the
    output is scaled to be in [-1, 1]."""

    def __init__(
        self,
        segment_duration: float = 1.0,
        fs: float = 8000,
        n_fft: int = 1024,
        stft_hop: int = 256,
        n_mels: int = 256,
        f_min: float = 300.0,
        f_max: float = 4000.0,
        amin: float = 1e-5,
        dynamic_range: float = 80.0,
        scale: bool = True,
    ):
        """
        Parameters
        ----------
            segment_duration (float): Duration of the audio segment in seconds.
            fs (float): Sampling rate.
            n_fft (int): Number of FFT bins for the STFT.
            stft_hop (int): Hop size between consecutive frames of the STFT.
            n_mels (int): Number of Mel bands.
            f_min (float): Minimum frequency of the Mel bands.
            f_max (float): Maximum frequency of the Mel bands.
            amin (float): Minimum amplitude of the Mel-spectrogram.
            dynamic_range (float): Dynamic range of the Mel-spectrogram in dB.
            scale (bool) : If True, scale the melspectrogram to be in [-1, 1] using the dynamic range.
        """

        super().__init__()

        assert (
            segment_duration > 0
        ), f"segment_duration should be positive but is {segment_duration}"
        assert fs > 0, f"fs should be positive but is {fs}"
        assert n_fft > 0, f"n_fft should be positive but is {n_fft}"
        assert stft_hop > 0, f"stft_hop should be positive but is {stft_hop}"
        assert n_mels > 0, f"n_mels should be positive but is {n_mels}"
        assert f_min > 0, f"f_min should be positive but is {f_min}"
        assert f_max > 0, f"f_max should be positive but is {f_max}"
        assert amin > 0, f"amin should be positive but is {amin}"
        assert (
            dynamic_range > 0
        ), f"dynamic_range should be positive but is {dynamic_range}"

        # Save the parameters
        self.segment_duration = segment_duration
        self.segment_len = int(fs * segment_duration)  # Convert to samples
        self.fs = fs
        self.n_fft = n_fft
        self.stft_hop = stft_hop
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.amin = amin
        self.dynamic_range = dynamic_range
        self.scale = scale

        # Create the frame generator
        self.frame_generator = lambda x: es.FrameGenerator(
            x,
            frameSize=n_fft,
            hopSize=stft_hop,
            startFromZero=False,  # Zero-center the first window
            lastFrameToEndOfFile=True,  # Center the last window TODO
            validFrameThresholdRatio=0,  # Pad the start and end of the audio
        )

        # Create the window
        self.window = es.Windowing(
            type="hann",
            normalized=False,  # TODO: Seems like tf is not normalized
            size=n_fft,
            symmetric=False,  # Not interested in phase
            zeroPhase=False,  # Not interested in phase
        )

        # Define the FFT
        self.spec = es.Spectrum(size=n_fft)

        # Define the Mel bands
        self.mb = es.MelBands(
            highFrequencyBound=f_max,
            inputSize=n_fft // 2 + 1,
            log=False,
            lowFrequencyBound=f_min,
            normalize="unit_tri",
            numberBands=n_mels,
            sampleRate=fs,
            type="magnitude",
            warpingFormula="slaneyMel",
            weighting="linear",
        )

    def compute(self, audio: np.ndarray) -> np.ndarray:
        """Compute the Power mel-spectrogram of an audio segment. The audio segment is padded
        from both sides to center the first window. The Mel-spectrogram is computed using
        Essentia, which only works with 'single' datatype (np.float32). If specified, the
        output is scaled to be in [-1, 1].

        Parameters:
        -----------
            audio (np.ndarray): Input audio of shape (T,). Converted to float32.

        Returns:
        --------
            mel_spec (np.ndarray): Power mel-spectrogram of shape (F, T) and dtype float32.
        """

        assert type(audio) == np.ndarray, "audio should be a numpy array"
        assert audio.ndim == 1, f"Input shape is {audio.shape} but should be (T,)"
        assert len(audio) == self.segment_len, (
            f"Expected input len is {self.segment_len} " f"but recieved {len(audio)}"
        )

        # Ensure float32 for essentia
        audio = audio.astype(np.float32)

        # Calculate the Magnitude Mel-spectrogram
        mel_spec = [
            self.mb(self.spec(self.window(frame)))
            for frame in self.frame_generator(audio)
        ]
        mel_spec = np.array(mel_spec)  # (n_frames, n_mels)

        # Clip magnitude below amin. This is to avoid log(0) in the next step
        mel_spec = np.where(mel_spec > self.amin, mel_spec, self.amin)

        # Convert to db scale, using max as the reference
        mel_spec = 20 * np.log10(mel_spec / np.max(mel_spec))
        # Allow a maximum of -dynamic_range dB dynamic range
        mel_spec = np.where(
            mel_spec > -self.dynamic_range, mel_spec, -self.dynamic_range
        )

        # Scale x to be in [-1, 1] if scale is True
        if self.scale:
            mel_spec = 1 + (mel_spec / (self.dynamic_range / 2))

        return mel_spec.T  # (n_mels, n_frames)

    def compute_batch(self, batch) -> np.ndarray:
        """Computes the Mel-spectrogram of a batch of audio segments.

        Parameters:
        ----------
            batch: (B, T)

        Returns:
        -------
            batch_mel: (B, F, T)
        """

        batch_mel = np.array([self.compute(audio) for audio in batch])

        return batch_mel

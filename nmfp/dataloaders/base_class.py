from tensorflow.keras.utils import Sequence

from nmfp.audio_processing.melspectrogram import Melspec_layer


class DataLoader(Sequence):
    """Base class for all data loading classes. It is a subclass of
    tensorflow.keras.utils.Sequence"""

    def __init__(
        self,
        segment_duration=1,
        hop_duration=0.5,
        fs: float = 8000,
        n_fft=1024,
        stft_hop=256,
        n_mels=256,
        f_min=300,
        f_max=4000,
        scale_output=True,
    ):
        """
        Parameters
        ----------
            segment_duration : (float), optional
                Segment duration in seconds. The default is 1.
            hop_duration : (float), optional
                Hop-size of segments in seconds. The default is .5.
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
        """

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
        self.scale_output = scale_output

        # Essentia melspec layer
        self.mel_spec = Melspec_layer(
            segment_duration=segment_duration,
            fs=fs,
            n_fft=n_fft,
            stft_hop=stft_hop,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            scale=scale_output,
        )

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

"""This module contains functions for degrading audio signals with
background noise and room impulse responses."""

from typing import Tuple

import numpy as np
from scipy.signal import convolve

from .normalization import peak_normalize, rms_energy

#### Utility Functions ####


def sample_random_number_batch(n: int, range: list, log_scale: bool) -> np.ndarray:
    """Sample n uniformly random numbers from range. Optionally, sample
    from a log scale.

    Parameters
    ----------
        n : int
            Number of samples to sample.
        range : tuple (float)
            SNR range in dB. (min, max)
        log_scale : bool
            If True, sample from a log scale.

    Returns
    -------
        random_number : np.ndarray
            Random number for each sample in the batch. (n,)
    """

    assert n > 0, "n should be greater than 0"
    assert len(range) == 2, "range should be (min, max)"
    assert range[0] < range[1], "range should be (min, max)"

    if log_scale:
        # Make sure log10(0) is avoided
        assert range[0] > 0, "range[0] should be positive if log_scale=True"

        log_min, log_max = np.log10(range)
        random_number_log = np.random.rand(n) * (log_max - log_min) + log_min
        random_number = np.power(10, random_number_log)
    else:
        min_snr, max_snr = range
        random_number = np.random.rand(n) * (max_snr - min_snr) + min_snr

    return random_number


def apply_random_gain_batch(
    x_batch: np.ndarray, gain_range: list = [0.1, 1.0]
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a random gain to each sample in the batch. The gain is
    sampled uniformly from a log scale within the gain_range.

    Parameters
    ----------
        x_batch : (np.ndarray)
            Batch of signals. (B, T)
        gain_range : (list)
            Range of gain values. (min, max)

    Returns
    -------
        x_batch_gain : (np.ndarray)
            Batch of signals with random gain. (B, T)
        gain : (np.ndarray)
            Random gain for each sample in the batch. (B,)
    """

    assert len(x_batch.shape) == 2, "x_batch should be a 2D array."
    assert gain_range[0] > 0, "gain_range[0] should be positive"
    assert gain_range[1] <= 1, "gain_range[1] cannot be greater than 1"
    assert gain_range[0] < gain_range[1], "gain_range should be (min, max)"

    # Random gain for each sample in the batch
    gain = sample_random_number_batch(
        n=x_batch.shape[0], range=gain_range, log_scale=True
    )

    # Apply the gains to each element in the batch independently
    x_batch_gain = x_batch * gain[:, np.newaxis]

    return x_batch_gain, gain


#### Background Noise Degradation ####


def background_mix(x: np.ndarray, x_bg: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix input signal with background noise with a specified SNR (dB).
    The returned signal has the same length as x. The input signals are
    RMS normalized before the mixture. Output signal is max-normalized.

    Parameters
    ----------
        x : 1D array (float)
            Input audio signal.
        x_bg : 1D array (float)
            Background noise signal.
        snr_db : (float)
            signal-to-noise ratio in decibel.

    Returns
    -------
        x_mix : 1D array
            Max-normalized mix of x and x_bg with SNR

    """

    assert len(x.shape) == 1, "x should be 1D."
    assert len(x_bg.shape) == 1, "x_bg should be 1D."
    assert len(x) == len(
        x_bg
    ), f"x=({len(x)}) and x_bg=({len(x_bg)}) should have the same length."

    # Get the RMS energy for each signal
    rms_bg = rms_energy(x_bg)
    rms_x = rms_energy(x)

    # mix based on the RMS energy of each signal
    if rms_bg != 0 and rms_x != 0:
        # Normalize each signal by its RMS Amplitude
        x_bg_norm = x_bg / rms_bg
        x_norm = x / rms_x
        # Mix with snr_db
        magnitude = np.power(10, snr_db / 20.0)
        x_mix = magnitude * x_norm + x_bg_norm
    elif rms_bg == 0 and rms_x == 0:
        # Both signals are zero so just return zeros
        x_mix = np.zeros_like(x)
        print("Both signals are zero!")
    elif rms_bg == 0:
        # One of the signal is zero so just add them
        x_mix = x + x_bg
        print("Noise signal is zero!")
    else:
        # One of the signal is zero so just add them
        x_mix = x + x_bg
        print("Input signal is zero!")

    # Max normalize the mix signal to avoid clipping
    x_mix = peak_normalize(x_mix)

    return x_mix


def bg_mix_batch(
    music_batch: np.ndarray, bg_batch: np.ndarray, snr_range=[0, 10]
) -> Tuple[np.ndarray, np.ndarray]:
    """Mix a batch of events with a batch of background noise with a uniformly
    random SNR (dB) from snr_range. A random SNR is sampled for each
    sample in the batch.

    Parameters
    ----------
        music_batch : (np.ndarray)
            Batch of music signals. (B, T)
        bg_batch : (np.ndarray)
            Batch of background noise signals. (B, T)
        snr_range : (list)
            SNR range in dB. (min, max)

    Returns
    -------
        X_bg_mix : (np.ndarray)
            Batch of mixed signals. (B, T)
        snrs : (np.ndarray)
            Random SNR for each sample in the batch. (B,)
    """

    assert len(snr_range) == 2, "snr_range should be (min, max)"
    assert snr_range[0] < snr_range[1], "snr_range should be (min, max)"
    assert (
        music_batch.shape == bg_batch.shape
    ), "music_batch and bg_batch should have the same shape."

    # Initialize
    X_bg_mix = np.zeros((music_batch.shape[0], music_batch.shape[1]))

    # Random SNR for each sample in the batch
    snrs = sample_random_number_batch(
        n=music_batch.shape[0], range=snr_range, log_scale=False
    )

    # Mix each element with random SNR
    for i in range(len(music_batch)):
        X_bg_mix[i] = background_mix(x=music_batch[i], x_bg=bg_batch[i], snr_db=snrs[i])

    return X_bg_mix, snrs


#### Room IR Degradation ####


def convolve_with_IR(x: np.ndarray, x_ir: np.ndarray) -> np.ndarray:
    """Convolve an input signal with an impulse response. The returned signal
    will have the same length as the input and it will be left aligned to
    preserve the original signal as much as possible without introducing
    x_ir length dependent delay. Finally, it is peak-normalized.

    Parameters
    ----------
        x : (np.ndarray)
            Input audio signal. 1D array.
        x_ir : (np.ndarray)
            Impulse response. 1D array.

    Returns
    -------
        x_deg : 1D array
            Max-normalized, left-aligned convolution of x and x_bg
    """

    assert len(x) > 0, "x should not be empty."
    assert len(x_ir) > 0, "x_ir should not be empty."
    assert type(x) == np.ndarray, "x should be a numpy array."
    assert type(x_ir) == np.ndarray, "x_ir should be a numpy array."
    assert len(x.shape) == 1, "x should be 1D."
    assert len(x_ir.shape) == 1, "x_ir should be 1D."

    # Display warning if any signal is all zeros
    if not np.any(x_ir):
        print("x_ir is all zeros!")
    if not np.any(x):
        print("x is all zeros!")

    # Convolve with impulse response
    x_deg = convolve(x, x_ir, mode="full", method="auto")
    # Trim the signal to the original length with left alignment
    x_deg = x_deg[: len(x)]
    # Max normalize the mix signal to avoid clipping
    x_deg = peak_normalize(x_deg)

    return x_deg


def convolve_with_IR_batch(X_batch: np.ndarray, ir_batch: list) -> np.ndarray:
    """Convolve a batch of audio with a batch of impulse responses.
    Uses convolve_with_IR() to convolve each element with the
    corresponding impulse response.

    Parameters
    ----------
        X_batch  : (np.ndarray)
            Batch of audio signals. (B, T)
        ir_batch : (list)
            Batch of impulse responses. (B, T')

    Returns
    -------
        X_ir_deg : (np.ndarray)
            Batch of degraded signals. (B, T)
    """

    assert len(X_batch.shape) == 2, "X_batch should be a 2D array."
    assert type(ir_batch) == list, f"ir_batch should be a list, not {type(ir_batch)} "
    assert X_batch.shape[0] == len(
        ir_batch
    ), "X_batch and ir_batch should have the same number of samples."

    # Initialize
    X_ir_deg = np.zeros_like(X_batch)

    # Convolve each element with the corresponding impulse response
    for i in range(X_batch.shape[0]):
        x = X_batch[i]
        x_ir = ir_batch[i]
        x_deg = convolve_with_IR(x, x_ir)
        X_ir_deg[i] = x_deg

    return X_ir_deg

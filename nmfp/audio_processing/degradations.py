"""This module contains functions for degrading audio signals with
background noise and room impulse responses."""

from typing import Tuple

import numpy as np


from .normalization import peak_normalize_batch, rms_normalize_batch

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


def bg_mix_batch(
    music: np.ndarray, bg: np.ndarray, snr_range=[0, 10]
) -> Tuple[np.ndarray, np.ndarray]:
    """Mix a batch of events with a batch of background noise with a uniformly
    random SNR (dB) from snr_range. A random SNR is sampled for each
    sample in the batch.

    Parameters
    ----------
        music : (np.ndarray)
            Batch of music signals. (B, T)
        bg : (np.ndarray)
            Batch of background noise signals. (B, T)
        snr_range : (list)
            SNR range in dB. (min, max)

    Returns
    -------
        mix : (np.ndarray)
            Batch of mixed signals. (B, T)
        snrs : (np.ndarray)
            Random SNR for each sample in the batch. (B,)
    """

    assert len(snr_range) == 2, "snr_range should be (min, max)"
    assert snr_range[0] < snr_range[1], "snr_range should be (min, max)"
    assert music.shape == bg.shape, "music and bg should have the same shape."

    # Random SNR for each sample in the batch
    snrs = sample_random_number_batch(
        n=music.shape[0], range=snr_range, log_scale=False
    )
    # Compute mixing magnitude for each sample
    magnitude = np.power(10, snrs[:, None] / 20.0)

    # RMS normalize the signals
    music = rms_normalize_batch(music)
    bg = rms_normalize_batch(bg)

    # Mix signals
    mix = magnitude * music + bg

    # Max normalize the mix signal to avoid clipping
    mix = peak_normalize_batch(mix)

    return mix, snrs


#### Room IR Degradation ####


def convolve_with_IR_batch(X_batch: np.ndarray, IR_batch: np.ndarray) -> np.ndarray:
    """Convolve a batch of audio signlas with a batch of impulse responses, row-by-row.
    Expects the same number of samples in both batches.

    Parameters
    ----------
        X_batch  : (np.ndarray)
            Batch of audio signals. (B, T)
        IR_batch : (np.ndarray)
            Batch of impulse responses. (B, L)

    Returns
    -------
        X_ir_deg : (np.ndarray)
            Batch of degraded signals. (B, T)
    """

    assert len(X_batch.shape) == 2, "X_batch should be a 2D array."
    assert (
        X_batch.shape[0] == IR_batch.shape[0]
    ), "X_batch and IR_batch should have the same number of samples."

    _, T = X_batch.shape
    _, L = IR_batch.shape
    N = T + L - 1  # Length after full convolution

    # Compute FFT along time axis.
    X_fft = np.fft.rfft(X_batch, n=N, axis=1)
    IR_fft = np.fft.rfft(IR_batch, n=N, axis=1)

    # Multiply in the frequency domain.
    Y_fft = X_fft * IR_fft

    # Inverse FFT to obtain time domain result.
    Y = np.fft.irfft(Y_fft, n=N, axis=1)

    # NOTE this should have came after the peak norm, but the
    # peak is supposed to happen much before the tail anyways
    # Trim to original length T.
    Y_trimmed = Y[:, :T]

    # Max normalize the mix signal to avoid clipping
    X_ir_deg = peak_normalize_batch(Y_trimmed)

    return X_ir_deg

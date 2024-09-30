""" Script for processing audio signals in terms on energy. """

import numpy as np


def rms_energy(x: np.ndarray) -> float:
    """Compute the RMS energy of an audio signal.

    Parameters
    ----------
        x : (np.ndarray)
            Input audio signal. Shape: (n_samples,)

    Returns
    -------
        rms_energy : (float)
            RMS energy of the input signal.
    """

    assert len(x.shape) == 1, "x should be 1D."

    rms_energy = np.sqrt(np.mean(x**2))

    return rms_energy


def rms_normalize(x: np.ndarray) -> np.ndarray:
    """RMS-normalize an audio signal. If the RMS energy of the
    signal is 0, the signal is returned as is.

    Parameters
    ----------
        x : (np.ndarray)
            Input audio signal. Shape: (n_samples,)

    Returns
    -------
        x_norm : (np.ndarray)
            RMS-normalized audio signal.
    """

    assert len(x.shape) == 1, "x should be 1D."

    # Get the RMS energy for the signal
    rms = rms_energy(x)

    # Normalize the signal by its RMS Energy
    if rms == 0:
        # X must be 0 everywhere
        x_norm = x
    else:
        x_norm = x / rms

    return x_norm


def peak_normalize(x: np.ndarray) -> np.ndarray:
    """Peak normalize an audio signal. If the maximum value of the
    signal is 0, the signal is returned as is.

    Parameters
    ----------
        x : (np.ndarray)

    Returns
    -------
        x_norm: (float)
            Max-normalized audio signal.
    """

    assert len(x.shape) == 1, "x should be 1D."

    # Get the rectified maximum value of the signal
    max_val = np.max(np.abs(x))

    # Normalize the signal by its maximum value
    if max_val == 0:
        # X must be 0 everywhere
        x_norm = x
    else:
        x_norm = x / max_val

    return x_norm

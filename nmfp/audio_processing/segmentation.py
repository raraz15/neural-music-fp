""" Script for segmenting audio signals, taking chunks, or reconstructing a segmented signal. """

from typing import Tuple

import numpy as np


def number_of_segments(
    signal_length: int, L: int, H: int, discard_remainder: bool = True
) -> Tuple[int, int]:
    """Calculates how many segments can be taken from an audio signal with L and
    H. By default discards the remainder. The window is not centered. If the signal is
    too short to segment, and discard_remainder is true, raises an error.

    Parameters
    ----------
        signal_length : (int)
            Length of the audio signal.
        L : (int)
            Length of the segments.
        H : (int)
            Hop size between segments.
        discard_remainder : (bool)
            If True, discards the remainder segment. If False, pads the remainder
            segment.

    Returns
    -------
        N : (int)
            Number of segments.
        remainder : (int)
            Number of samples in the remainder segment.
    """

    assert L > 0, "L should be positive"
    assert H > 0, "H should be positive"
    assert H <= L, "H should be smaller than or equal to L"

    if signal_length < L:
        assert not discard_remainder, (
            "signal_length is too short for L. "
            "Discarding the remainder segment would result in 0 segments."
        )

        N = 1  # Only the remainder segment
        remainder = signal_length

    else:
        # Number of complete segments
        N = (signal_length - L) // H + 1

        # Check for remainder
        remainder = signal_length - ((N - 1) * H + L)
        assert remainder >= 0, "remainder can not be negative."

        # If we have a remainder and we don't want to discard it, add it to the number of segments
        if remainder > 0 and not discard_remainder:
            N += 1

    return N, remainder


def test_number_of_segments():
    N, remainder = number_of_segments(
        signal_length=27, L=8000, H=4000, discard_remainder=False
    )
    assert N == 1 and remainder == 27, "Test 1 failed"

    # Not sure about his one
    N, remainder = number_of_segments(
        signal_length=8027, L=8000, H=4000, discard_remainder=False
    )
    assert N == 2 and remainder == 27, "Test 2 failed"

    N, remainder = number_of_segments(
        signal_length=8027, L=8000, H=4000, discard_remainder=True
    )
    assert N == 1 and remainder == 27, "Test 2 failed"

    N, remainder = number_of_segments(
        signal_length=12027, L=8000, H=4000, discard_remainder=False
    )
    assert N == 3 and remainder == 27, "Test 3 failed"

    N, remainder = number_of_segments(
        signal_length=12027, L=8000, H=4000, discard_remainder=True
    )
    assert N == 2 and remainder == 27, "Test 3 failed"


def segment_audio(
    audio: np.ndarray, L: int, H: int, discard_remainder: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Cut the audio into consecutive segments of segment length L and hop
    size H. By default discards the remainder segment. If specified, pads the
    remainder segment.

    Parameters
    ----------
        audio : (np.ndarray)
            Input audio signal. Shape: (n_samples,)
        L : (int)
            Length of the segments.
        H : (int)
            Hop size between segments.
        discard_remainder : (bool)
            If True, discards the remainder segment. If False, pads the remainder
            segment.

    Returns
    -------
        segments : (np.ndarray)
            Segmented audio signal. Shape: (N, L)
        boundaries : (np.ndarray)
            Boundaries of the segments. Shape: (N, 2)
    """

    assert type(audio) == np.ndarray, "audio should be a numpy array"
    assert len(audio.shape) == 1, "audio should be 1D array"

    # Calculate the number of segments that can be cut from the audio
    N_cut, _ = number_of_segments(len(audio), L, H, discard_remainder)

    # Initialize the segmented output
    segments = np.zeros((N_cut, L))

    boundaries = []
    for i in range(N_cut):
        start = i * H
        end = start + L
        boundaries.append([start, end])
        # If we have enough samples for a full window, copy it over
        if end <= len(audio):
            segments[i, :] = audio[start:end]
        # If we're at the end and need to pad the remainder
        elif not discard_remainder:
            # The rest of the window remains zero-padded
            segments[i, : len(audio) - start] = audio[start:]
            # Update the end boundary
            boundaries[i][1] = len(audio)
        else:
            # If we don't want to pad, we just discard the last segment
            segments = segments[:i, :]
            boundaries = boundaries[:i]
            break
    # Convert to numpy array
    boundaries = np.array(boundaries)

    assert boundaries[-1, -1] <= len(audio), (
        f"The last boundary {boundaries[-1,-1]} is larger than "
        f"the length of the audio {len(audio)}"
    )

    return segments, boundaries


def get_random_chunk(audio: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a random chunk of N samples from the audio signal.

    Parameters
    ----------
        audio : (np.ndarray)
            Input audio signal. Shape: (n_samples,)
        N : (int)
            Length of the chunk.

    Returns
    -------
        chunk : (np.ndarray)
            Random chunk of N samples from the audio signal.
    """

    assert type(audio) == np.ndarray, "audio should be a numpy array"
    assert len(audio.shape) == 1, "audio should be 1D array"
    assert N > 0, "N must be positive"

    # Get a random start index
    start = np.random.randint(0, len(audio) - N)
    end = start + N
    # Get the boundaries of the chunk
    boundary = np.array([start, end])

    # Copy the chunk to avoid modifying the original audio
    chunk = audio[start:end].copy()

    return chunk, boundary


def OLA(segments: np.ndarray, overlap_ratio: float) -> np.ndarray:
    """Overlap and add segments."""

    # Check inputs
    assert len(segments.shape) == 2, "segments should be 2D array"
    assert (
        overlap_ratio >= 0 and overlap_ratio <= 1
    ), "overlap_ratio should be between 0 and 1"
    if overlap_ratio != 0.5:
        raise NotImplementedError("We only support overlap_ratio=0.5 for now.")
    assert len(segments.shape) == 2, "segments should be 2D array"

    # Get the number of segments and samples
    n_segments, n_samples = segments.shape

    # Calculate the hop size and the number of samples in the output
    hop = int(n_samples * (1 - overlap_ratio))
    n_samples_out = (n_segments - 1) * hop + n_samples

    out = np.zeros(n_samples_out)
    for i in range(n_segments):
        out[i * hop : i * hop + n_samples] += segments[i]

    # Since we are adding the segments, we need to divide the overlapping parts by 2
    out[hop:-hop] /= 2

    return out

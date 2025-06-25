import os
import numpy as np

import soundfile as sf

from .normalization import peak_normalize


def load_wav(
    file_path: str,
    seg_start_sec: float = 0.0,
    offset_sec: float = 0.0,
    seg_dur_sec: float = None,
    seg_pad_offset_sec=0.0,
    fs: float = 8000,
    normalize: bool = False,
    pad_if_short: bool = False,
) -> np.ndarray:
    """Opens a wav file, checks its format and sample rate, and returns the specified segment.
    Uses SoundFile for fast I/O.

    Parameters:
    -----------
        file_path (string): Must be a wav file.
        seg_start_sec: Start time of the segment in seconds.
        offset_sec: Additional offset from seg_start_sec.
        seg_dur_sec: Length of the segment in seconds (None to read to end of file).
        seg_pad_offset_sec: If padding is required, pad the segment starting at this offset (in seconds).
        fs (float): Expected sample rate.
        normalize: Peak-normalize the audio signal.
        pad_if_short: If the segment is shorter than seg_dur_sec, pad with zeros.

    Returns:
    --------
        x (np.ndarray) : Audio segment as a 1D array of floats in the range [-1, 1].
    """
    assert seg_start_sec >= 0.0, "The start time must be positive"
    if seg_dur_sec is not None:
        assert seg_dur_sec > 0.0, "If you specify a duration, it must be positive."

    # Check file extension
    file_ext = os.path.splitext(file_path)[1]
    assert file_ext == ".wav", f"Only .wav files are supported. Got {file_ext}"

    # Get file info using SoundFile
    info = sf.info(file_path)

    _fs = info.samplerate
    if fs != _fs:
        raise ValueError(f"Sample rate should be {fs} but got {_fs} for {file_path}")
    n_total_samples = info.frames
    assert n_total_samples > 0, f"{file_path} has no samples."

    # Calculate starting sample and desired segment length
    start_sample = int(np.floor((seg_start_sec + offset_sec) * fs))
    if seg_dur_sec is None:
        seg_length = n_total_samples - start_sample
    else:
        seg_length = int(np.floor(seg_dur_sec * fs))

    # Read the segment using SoundFile
    x, sr = sf.read(file_path, start=start_sample, frames=seg_length, dtype="float32")
    # Note: When reading with dtype='float32', SoundFile scales PCM data to [-1, 1]

    # Pad if needed
    if pad_if_short and len(x) < seg_length:
        if len(x) < seg_length // 2:
            print(
                f"Warning: {file_path} is shorter than half of the segment duration. Padding with zeros."
            )
        padded = np.zeros(seg_length, dtype=np.float32)
        seg_pad_offset_idx = int(seg_pad_offset_sec * fs)
        assert (
            seg_pad_offset_idx + len(x) <= seg_length
        ), "The padded segment is longer than input duration and seg_pad_offset_sec."
        padded[seg_pad_offset_idx : seg_pad_offset_idx + len(x)] = x
        x = padded

    # Normalize if requested
    if normalize:
        x = peak_normalize(x)

    return x


def write_wav(
    output_path: str, x: np.ndarray, fs: float = 8000, normalize: bool = False
) -> None:
    """Expects a 1D float numpy array and writes it a 16-bit integer LPCM wav file.

    NOTE: I had used the wave module from python standard library before to write 
    all the wav files but in the last few months, I started using SoundFile for 
    READING wav files, but for writing wav files (while creating the dataset) I 
    used the wave library. In fact, I haven't tested the last line of this method 
    which uses SoundFile yet."""

    assert x.ndim == 1, "Only 1D arrays are supported."
    assert x.dtype.type in {
        np.float64,
        np.float32,
    }, "Only float64 or float32 arrays are supported."

    # Max Normalize if specified
    if normalize:
        x = peak_normalize(x)

    # Maximum amplitude allowed with 16-bit PCM
    amplitude = np.iinfo(np.int16).max
    x = (x * amplitude).astype(np.int16)

    # Write the audio file
    sf.write(output_path, x, int(fs), subtype="PCM_16")

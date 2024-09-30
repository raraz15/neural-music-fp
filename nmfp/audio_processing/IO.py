import os
import numpy as np
import wave

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
    Uses native python wave module and numpy.

    Parameters:
    -----------
        file_path (string): Must be a wav file with 16-bit LPCM encoding.
        seg_start_sec: Start of the segment in seconds.
        offset_sec: Offset from seg_start_sec in seconds.
        seg_dur_sec: Length of the segment in seconds.
        seg_pad_offset_sec: If padding is required (seg_dur_sec is longer than file duration),
            pad the segment from the rgiht and give an offset from the left
            with this amount of seconds.
        fs (float): sample rate
        normalize: peak-normalize the audio signal
        pad_if_short: If there is a remainder segment, and
         it is shorter than seg_dur_sec, pad it with zeros.

    Returns:
    --------
        x (np.ndarray) : Audio segment. 1D array of shape (T,) with dtype float64 'double'.

    """

    assert seg_start_sec >= 0.0, "The start time must be positive"
    if seg_dur_sec is not None:
        assert seg_dur_sec > 0.0, (
            "If you specify a duration, it must be positive."
            "Use None to read the rest of the file."
        )

    # Check file extension
    file_ext = os.path.splitext(file_path)[1]
    assert file_ext == ".wav", f"Only .wav files are supported. Got {file_ext}"

    # Open file
    pt_wav = wave.open(file_path, "r")

    # Check sample rate
    _fs = pt_wav.getframerate()
    if fs != _fs:
        raise ValueError(f"Sample rate should be {fs} but got {_fs} for {file_path}")

    # Get the number of samples the file has if requested
    n_total_samples = pt_wav.getnframes()
    assert n_total_samples > 0, f"{file_path} has no samples."

    # Calculate segment start index
    start_sample = np.floor((seg_start_sec + offset_sec) * fs).astype(int)
    pt_wav.setpos(start_sample)

    # Determine the segment length
    if seg_dur_sec is None:
        # if seg_dur_sec is None, read the rest of the file
        seg_length = n_total_samples - start_sample
    else:
        # if seg_dur_sec is bigger than the file size, it loads everything.
        seg_length = np.floor(seg_dur_sec * fs).astype(int)

    # Load audio and close file
    x = pt_wav.readframes(seg_length)
    pt_wav.close()

    # Convert bytes to float
    x = np.frombuffer(x, dtype=np.int16)
    x = x / np.iinfo(np.int16).max  # scale to [-1, 1] float

    # If specified and the segment is shorter than seg_dur_sec, pad it
    if pad_if_short and len(x) < seg_length:
        # Warn if the segment is shorter than half of the segment duration since
        # the padded segment will be mostly zeros
        if len(x) < seg_length // 2:
            print(
                f"Warning: {file_path} is shorter than half of the segment duration. "
                f"Padding with zeros."
            )
        _x = np.zeros(int(seg_dur_sec * fs))
        # Start the signal from seg_pad_offset_sec
        seg_pad_offset_idx = int(seg_pad_offset_sec * fs)
        assert (
            seg_pad_offset_idx + len(x) <= seg_length
        ), "The padded segment is longer than input duration and seg_pad_offset_sec."
        _x[seg_pad_offset_idx : seg_pad_offset_idx + len(x)] = x
        x = _x

    # Max Normalize if specified
    if normalize:
        x = peak_normalize(x)

    return x


def write_wav(
    output_path: str, x: np.ndarray, fs: float = 8000, normalize: bool = False
) -> None:
    """Expects a 1D float numpy array and writes it a 16-bit integer LPCM wav file."""

    assert x.ndim == 1, "Only 1D arrays are supported."
    assert x.dtype.type in {
        np.float64,
        np.float32,
    }, "Only float64 or float32 arrays are supported."

    # Max Normalize if specified
    if normalize:
        x = peak_normalize(x)

    # Maximum amplitude allowed with 16-bit LPCM
    amplitude = np.iinfo(np.int16).max
    # Convert x to 16-bit integers
    x = (x * amplitude).astype(np.int16)

    with wave.open(output_path, "w") as wav_file:
        # Set parameters:
        wav_file.setnchannels(1)  # 1 channel (mono)
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(fs)  # sample rate
        wav_file.setnframes(len(x))  # set the number of frames
        # Write
        wav_file.writeframes(x.tobytes())

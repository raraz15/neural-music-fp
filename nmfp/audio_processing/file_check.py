import os

import soundfile as sf

from .segmentation import number_of_segments


def check_wav_file(file_path: str, fs: float = 8000) -> int:
    """Checks if file_path is .wav and has the same sample rate. Can also return
    the number of samples the file has from reading it from the header.

    Parameters:
    -----------
        file_path (str) : path to audio file.
        fs (float) : sample rate.

    Returns:
    --------
        n_total_samples (int) : number of samples the file has.
    """

    # Only support .wav
    file_ext = os.path.splitext(file_path)[1]
    if file_ext != ".wav":
        raise NotImplementedError(f"Only .wav files are supported. Got {file_ext}")

    # Get file info using SoundFile
    info = sf.info(file_path)

    _fs = info.samplerate
    if fs != _fs:
        raise ValueError(f"Sample rate should be {fs} but got {_fs} for {file_path}")

    # Get the number of samples the file has if requested
    n_total_samples = info.frames
    assert n_total_samples > 0, f"{file_path} has no samples."

    return n_total_samples


def get_track_segment_dict(
    file_paths: list,
    fs: float = 8000,
    segment_duration: float = 1,
    hop_duration: float = 0.5,
    discard_remainder: bool = True,
    skip_short: bool = True,
) -> dict:
    """Opens an audio file, checks its format and sample rate, and creates a list
    of segments and possible offset ranges. The first segment starts from the
    beginning of the file. If there are residual samples at the end that can not
    form a segment of length segment_duration, they can be ignored. The first
    segment can not be offsetted to the left and the last segment can not be
    offsetted to the right.

    Parameters:
    -----------
        file_paths (list(str)) : list of track or chunk audio paths.
        fs (float) : sample rate.
        segment_duration (float) : segment duration in seconds.
        hop_duration (float) : hop duration in seconds.
        discard_remainder (bool) : if True, discard residual samples at the end.
        skip_short (bool) : if True, skip files that are shorter than the
            required segment duration.

    Returns:
    --------
        segment_dict (dict) :
            {audio_path: [[seg_idx, offset_min, offset_max], ...]}
                audio_path is a string
                seg_idx is an integer
                offset_min is 0 or negative integer
                offset_max is 0 or positive integer
    """

    # Just in case
    file_paths = sorted(file_paths)

    # Convert to samples
    segment_len = int(fs * segment_duration)
    hop_len = int(fs * hop_duration)

    # Create a dictionary of segments for each file
    segment_dict = {}
    for audio_path in file_paths:
        # Get the number of samples the file has
        n_total_samples = check_wav_file(audio_path, fs)

        # Skip files that are shorter than the required segment duration
        if skip_short and n_total_samples < segment_len:
            print(
                f"{audio_path} has {n_total_samples} samples,"
                " which is shorter than the required segment duration."
            )
            continue

        # Calculate the number of segments that can be cut from the audio
        # n_segs will always be at least 1 by our design.
        n_segs, _ = number_of_segments(
            n_total_samples,
            L=segment_len,
            H=hop_len,
            discard_remainder=discard_remainder,
        )

        # Create a list of segments from the file
        segment_dict[audio_path] = []
        # A segment should be able to randomly offsetted to the left
        # or right without going out of bounds completely.
        for seg_idx in range(n_segs):
            left_boundary = -hop_len
            right_boundary = hop_len
            if seg_idx == 0:  # first seg
                left_boundary = 0  # No samples to the left
            if seg_idx == n_segs - 1:  # last seg
                right_boundary = 0  # No samples to the right
            segment_dict[audio_path].append([seg_idx, left_boundary, right_boundary])

    return segment_dict

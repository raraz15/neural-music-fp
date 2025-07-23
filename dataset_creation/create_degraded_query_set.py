"""Creates a query dataset for testing by sampling audio chunks from the
specified .wav files. Each chunk is randomly shifted in time to simulate
mismatch between the query moment and stored fingerprints. In practice, this is
not necessary, as the chunks are not sampled with respect to the segments. But we
did some preliminary experiments regarding this, so it is kept. The chunks and their
degraded versions are saved as .wav files and their start and end indices in the original
audio file are saved as a .npy file. We do not use multiprocessing here to
preserve reproducibility."""

import os
import sys
import argparse
import glob
from typing import Tuple, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nmfp.dataloaders.loaders import DegradationLoader
from nmfp.audio_processing import write_wav, check_wav_file
from nmfp.utils import set_seed


def get_degradation_loader(
    tracks_csv: str,
    tracks_dir: str,
    fs: float,
    bg_root_dir: str,
    room_ir_root_dir: str,
    mic_ir_root_dir: str,
    bg_snr: Tuple[float, float],
    pre_rir_random_gain_range: Tuple[float, float],
    pre_mir_random_gain_range: Tuple[float, float],
) -> DegradationLoader:
    """This method will create a dataloader that is intended for augmenting
    audio chunks. For more information about the parameters, please refer to
    the DegradationLoader class."""

    # Get the track ids from the df
    df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    track_ids = df.index.tolist()

    # Create the track paths and check if they exist in tracks_dir
    clean_query_paths = []
    for track_id in track_ids:
        track_path = os.path.join(tracks_dir, track_id[:3], track_id + ".wav")
        if os.path.isfile(track_path):
            clean_query_paths.append(track_path)
    assert len(clean_query_paths) == len(
        track_ids
    ), "Some tracks do not exist in the specified directory."

    # Read the degradation files
    if bg_root_dir == "":
        print("No background noise will be used for degradation.")
        bg_aug = False
        ts_bg_fps = []
    else:
        bg_aug = True
        ts_bg_fps = sorted(
            glob.glob(os.path.join(bg_root_dir, "**", "*.wav"), recursive=True)
        )
        print(f" BG: {len(ts_bg_fps):>6,}")
        assert len(ts_bg_fps) > 0, f"No background noise found in {bg_root_dir}."

    if room_ir_root_dir == "":
        print("No room impulse response will be used for degradation.")
        room_ir_aug = False
        ts_rir_fps = []
    else:
        ts_rir_fps = sorted(
            glob.glob(os.path.join(room_ir_root_dir, "**", "*.wav"), recursive=True)
        )
        print(f"RIR: {len(ts_rir_fps):>6,}")
        room_ir_aug = True
        assert (
            len(ts_rir_fps) > 0
        ), f"No room impulse response found in {room_ir_root_dir}."

    if mic_ir_root_dir == "":
        print("No microphone impulse response will be used for degradation.")
        mic_ir_aug = False
        ts_mir_fps = []
    else:
        ts_mir_fps = sorted(
            glob.glob(os.path.join(mic_ir_root_dir, "**", "*.wav"), recursive=True)
        )
        print(f"MIR: {len(ts_mir_fps):>6,}")
        mic_ir_aug = True
        assert (
            len(ts_mir_fps) > 0
        ), f"No microphone impulse response found in {mic_ir_root_dir}."

    # Collect the degradation parameters
    ts_bg_parameters = [
        bg_aug,
        ts_bg_fps,
        bg_snr,
    ]
    ts_rir_parameters = [
        room_ir_aug,
        ts_rir_fps,
        pre_rir_random_gain_range,
    ]
    ts_mir_parameters = [
        mic_ir_aug,
        ts_mir_fps,
        pre_mir_random_gain_range,
    ]

    # Create the degradation dataset
    loader = DegradationLoader(
        chunk_paths=clean_query_paths,
        fs=fs,
        bg_aug_parameters=ts_bg_parameters,
        room_ir_aug_parameters=ts_rir_parameters,
        mic_ir_aug_parameters=ts_mir_parameters,
        shuffle_aug=True,
    )

    return loader


def main(
    loader,
    output_dir: str,
    chunk_duration: float,
    max_shift_duration: float,
    fs: float,
    previous_chunks_dir: Optional[str],
):
    assert chunk_duration > 0, "chunk_duration must be positive."
    assert max_shift_duration > 0, "max_shift_duration must be positive."

    # Convert the durations to samples
    chunk_len = int(chunk_duration * fs)
    max_shift_len = int(max_shift_duration * fs)

    # log file for recording failed audio files
    log_path = os.path.join(output_dir, "log.txt")
    # Clear the log file if it exists
    if os.path.isfile(log_path):
        open(log_path, "w").close()

    # Create the output directories
    clean_chunk_root_dir = os.path.join(output_dir, "clean")
    print(f"Clean chunks will be written to {clean_chunk_root_dir}")

    shifted_chunk_root_dir = os.path.join(output_dir, "clean-time_shifted")
    print(f"Clean, time shifted chunks will be written to {shifted_chunk_root_dir}")

    degraded_chunk_root_dir = os.path.join(output_dir, "clean-degraded")
    print(f"Clean, degraded chunks will be written to {degraded_chunk_root_dir}")

    degraded_shifted_chunk_root_dir = os.path.join(
        output_dir, "clean-time_shifted-degraded"
    )
    print(
        f"Clean, time shifted, degraded chunks will be written to {degraded_shifted_chunk_root_dir}"
    )

    # Sample chunks from each audio file and process
    for i, audio_path in enumerate(loader.chunk_paths):
        # Get the audio name
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]

        # Create the output directories for the chunks
        audio_id = audio_name[:3]
        chunk_dir = os.path.join(clean_chunk_root_dir, audio_id)
        chunk_shifted_dir = os.path.join(shifted_chunk_root_dir, audio_id)
        chunk_degraded_dir = os.path.join(degraded_chunk_root_dir, audio_id)
        chunk_shifted_degraded_dir = os.path.join(
            degraded_shifted_chunk_root_dir, audio_id
        )
        os.makedirs(chunk_degraded_dir, exist_ok=True)
        os.makedirs(chunk_shifted_degraded_dir, exist_ok=True)

        # Define the paths to save the chunks
        chunk_path = os.path.join(chunk_dir, audio_name + ".wav")
        chunk_shifted_path = os.path.join(chunk_shifted_dir, audio_name + ".wav")
        chunk_degraded_path = os.path.join(chunk_degraded_dir, audio_name + ".wav")
        chunk_shifted_degraded_path = os.path.join(
            chunk_shifted_degraded_dir, audio_name + ".wav"
        )

        # Skip if the chunk already exists
        if (
            os.path.isfile(chunk_path)
            and os.path.isfile(chunk_shifted_path)
            and os.path.isfile(chunk_degraded_path)
            and os.path.isfile(chunk_shifted_degraded_path)
        ):
            print(f"Chunk already exists. Skipping {audio_name}.")
            continue

        # Check the audio file. get its length
        track_len = check_wav_file(audio_path, fs)

        # Load the full track and degrade it
        track, track_degraded = loader.__getitem__(i)
        assert len(track) == len(
            track_degraded
        ), "Track and degraded track should have the same length."

        # If this is the first degradation, sample a random shift duration
        if previous_chunks_dir is None:

            # Sample a random shift_duration
            shift_len = np.random.randint(-max_shift_len, max_shift_len)
            # Min duration to sample a chunk and a shifted version
            min_samples = chunk_len + np.abs(shift_len)

            # Skip if the audio is too short
            if track_len < min_samples:
                print(f"{audio_path} duration ({track_len/fs:.2f} sec) is too short. ")
                continue

            # Sample a random start index
            start = np.random.randint(0, track_len - min_samples)
            # Boundaries of the total chunk including the shift
            total_boundary = np.array([start, start + min_samples])

            # Get the chunks
            total_chunk = track[total_boundary[0] : total_boundary[1]]
            total_chunk_degraded = track_degraded[total_boundary[0] : total_boundary[1]]

            # Get the chunks and boundaries accordingly
            if shift_len > 0:
                chunk = total_chunk[:-shift_len]
                chunk_shifted = total_chunk[shift_len:]
                chunk_degraded = total_chunk_degraded[:-shift_len]
                chunk_shifted_degraded = total_chunk_degraded[shift_len:]
                chunk_boundary = np.array(
                    [total_boundary[0], total_boundary[1] - shift_len]
                )
                chunk_shifted_boundary = np.array(
                    [total_boundary[0] + shift_len, total_boundary[1]]
                )
            elif shift_len < 0:
                chunk = total_chunk[-shift_len:]
                chunk_shifted = total_chunk[:shift_len]
                chunk_degraded = total_chunk_degraded[-shift_len:]
                chunk_shifted_degraded = total_chunk_degraded[:shift_len]
                chunk_boundary = np.array(
                    [total_boundary[0] - shift_len, total_boundary[1]]
                )
                chunk_shifted_boundary = np.array(
                    [total_boundary[0], total_boundary[1] + shift_len]
                )
            else:
                chunk = total_chunk[total_boundary[0] : total_boundary[1]]
                chunk_shifted = chunk
                chunk_degraded = total_chunk_degraded[
                    total_boundary[0] : total_boundary[1]
                ]
                chunk_shifted_degraded = chunk_degraded
                chunk_boundary = total_boundary
                chunk_shifted_boundary = chunk_boundary

            # Check the lengths
            assert (
                len(chunk) == chunk_len
            ), f"Chunk length {len(chunk)} does not match the required length {chunk_len} in {audio_path}"
            assert (
                len(chunk_shifted) == chunk_len
            ), f"Shifted chunk length {len(chunk_shifted)} does not match the required length {chunk_len} in {audio_path}"

            # Check the boundaries
            assert (
                np.diff(chunk_boundary)[0] == chunk_len
            ), f"Chunk boundary {chunk_boundary} does not match the required length {chunk_len} in {audio_path}"
            assert (
                np.diff(chunk_shifted_boundary)[0] == chunk_len
            ), f"Shifted chunk boundary {chunk_shifted_boundary} does not match the required length {chunk_len} in {audio_path}"

            os.makedirs(chunk_dir, exist_ok=True)
            os.makedirs(chunk_shifted_dir, exist_ok=True)

            write_wav(chunk_path, chunk, fs=fs)
            write_wav(chunk_shifted_path, chunk_shifted, fs=fs)

            np.save(os.path.join(chunk_dir, audio_name + ".npy"), chunk_boundary)
            np.save(
                os.path.join(chunk_shifted_dir, audio_name + ".npy"),
                chunk_shifted_boundary,
            )

        else:  # If we are degrading the same tracks with another set of degradations

            # Read the originally saved boundary
            chunk_boundary = np.load(
                os.path.join(
                    previous_chunks_dir, "clean", audio_name[:3], audio_name + ".npy"
                )
            )
            chunk_shifted_boundary = np.load(
                os.path.join(
                    previous_chunks_dir,
                    "clean-time_shifted",
                    audio_name[:3],
                    audio_name + ".npy",
                )
            )

            chunk_degraded = track_degraded[chunk_boundary[0] : chunk_boundary[1]]
            chunk_shifted_degraded = track_degraded[
                chunk_shifted_boundary[0] : chunk_shifted_boundary[1]
            ]

        # Check the lengths
        assert (
            len(chunk_degraded) == chunk_len
        ), f"Degraded chunk length {len(chunk_degraded)} does not match the required length {chunk_len} in {audio_path}"
        assert (
            len(chunk_shifted_degraded) == chunk_len
        ), f"Shifted degraded chunk length {len(chunk_shifted_degraded)} does not match the required length {chunk_len} in {audio_path}"

        # Write the chunks and their boundaries to disk.

        write_wav(chunk_degraded_path, chunk_degraded, fs=fs)
        write_wav(chunk_shifted_degraded_path, chunk_shifted_degraded, fs=fs)

        np.save(os.path.join(chunk_degraded_dir, audio_name + ".npy"), chunk_boundary)
        np.save(
            os.path.join(chunk_shifted_degraded_dir, audio_name + ".npy"),
            chunk_shifted_boundary,
        )

        # Print progress
        if (i + 1) % 1000 == 0 or (i + 1) == len(loader) or i == 0:
            print(f"Processed [{i+1:>{len(str(len(loader)))}}/{len(loader)}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "tracks_csv", type=str, help="CSV file containing the query set tracks."
    )
    parser.add_argument(
        "tracks_dir", type=str, help="Root directory of the query set tracks."
    )
    parser.add_argument(
        "--previous_chunks_dir",
        type=str,
        default=None,
        help="Path to the directory containing the original music tracks. "
        "If provided, the chunks will be taken as the same boundaries "
        "as the original ones.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="Path to the root directory of the dataset. "
        "Sampled chunks will be written here inside the "
        "corresponding partition. Defaults to next to the paths_file."
        "Directory structure will be:\n"
        "output_dir/<partition>/xxx/xxx.wav",
    )
    parser.add_argument(
        "--chunk_duration",
        type=float,
        default=30.0,
        help="Duration of a sampled audio chunk in seconds.",
    )
    parser.add_argument(
        "--max_shift_duration",
        type=float,
        default=0.25,
        help="The maximum amount of time shift that will be applied "
        "to the clean chunk in seconds. This is done to simulate the "
        "real-life mismatch between the query moment and stored "
        "fingerprints.",
    )
    parser.add_argument(
        "--bg_root_dir",
        type=str,
        default="",
        help="Root directory of the background noise recordings.",
    )
    parser.add_argument(
        "--bg_snr_range",
        type=str,
        default="0,10",
        help="Range of SNR values for background noise degradation.",
    )
    parser.add_argument(
        "--rir_root_dir",
        type=str,
        default="",
        help="Root directory of the Room IR recordings.",
    )
    parser.add_argument(
        "--pre_rir_random_gain_range",
        type=str,
        default="0.1,1.0",
        help="Range of random gain values for pre Room IR degradation.",
    )
    parser.add_argument(
        "--mir_root_dir",
        type=str,
        default="",
        help="Root directory of the Microphone IR recordings.",
    )
    parser.add_argument(
        "--pre_mir_random_gain_range",
        type=str,
        default="0.1,1.0",
        help="Range of random gain values for pre Microphone IR degradation.",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=8000,
        help="Sampling frequency of the audio files.",
    )
    args = parser.parse_args()

    # Set the seed
    set_seed(seed_tf=False)

    # Check the degradation parameters
    assert args.bg_root_dir or args.rir_root_dir or args.mir_root_dir, (
        "At least one of the degradation parameters should be provided. "
        "Use --bg_root_dir, --rir_root_dir, or --mir_root_dir"
    )

    # Parse numeric degradation parameters
    args.bg_snr_range = sorted(tuple([float(x) for x in args.bg_snr_range.split(",")]))
    assert (
        len(args.bg_snr_range) == 2
    ), "bg_snr_range should have two values separated by a comma."
    args.pre_rir_random_gain_range = sorted(
        tuple([float(x) for x in args.pre_rir_random_gain_range.split(",")])
    )
    assert (
        len(args.pre_rir_random_gain_range) == 2
    ), "pre_rir_random_gain_range should have two values separated by a comma."
    args.pre_mir_random_gain_range = sorted(
        tuple([float(x) for x in args.pre_mir_random_gain_range.split(",")])
    )
    assert (
        len(args.pre_mir_random_gain_range) == 2
    ), "pre_mir_random_gain_range should have two values separated by a comma."

    # Determine the output directory and create it
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.paths_file)
    print(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Create the loader
    query_loader = get_degradation_loader(
        args.tracks_csv,
        args.tracks_dir,
        fs=args.fs,
        bg_root_dir=args.bg_root_dir,
        room_ir_root_dir=args.rir_root_dir,
        mic_ir_root_dir=args.mir_root_dir,
        bg_snr=args.bg_snr_range,
        pre_rir_random_gain_range=args.pre_rir_random_gain_range,
        pre_mir_random_gain_range=args.pre_mir_random_gain_range,
    )

    # Cut segments from each audio file and write them to disk
    main(
        query_loader,
        args.output_dir,
        args.chunk_duration,
        args.max_shift_duration,
        args.fs,
        previous_chunks_dir=args.previous_chunks_dir,
    )

    print("Done!")

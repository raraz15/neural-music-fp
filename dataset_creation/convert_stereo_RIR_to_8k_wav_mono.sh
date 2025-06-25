#!/bin/bash

#########################################################################################

if [ $# == 0 ]; then
    echo "Description: A simple script to recursively resample a bunch of files
          in a directory. Only certain file extensions (aac, flac, wav) are considered.

          It takes 2 command line options: indir and outdir.
          The destination (outdir) is relative to the current
          directory of where you were when the script was run.
          16-bit LPCM mono WAV files are written to outdir. Each stereo
          file is converted to 2 mono files, one for each channel.

          Example: resample.sh audio/ resampled/

          The directory structure inside indir will be replicated
          in outdir."
    echo "Usage: $0 param1 param2"
    echo "param1: Input directory"
    echo "param2: Output directory"
    exit 0
fi

#########################################################################################

# Sourece directory with files to convert
InDir=$1

# Set the directory you want for the converted files
OutDir=$2

mkdir -p "$OutDir"

# Target sample rate
TARGET_SR=8000

# start a log file
LogFile=$OutDir"/sox_log.txt"

# remove the log file if it already exists
rm -f $LogFile

# Convert each file with SoX, and write the converted file
# to the corresponding output dir, preserving the internal
# structure of the input dir
find $InDir -regextype posix-extended -type f -iregex '.*\.(wav|flac|aac)$' -print0 | while read -d $'\0' input
do

    echo "processing" $input |& tee -a $LogFile

    # the output path, without the InDir prefix
    output=${input#$InDir}

    # remove the extension
    output=${output%.*}

    # replace the original extension with .wav and add OutDir
    # also 2 paths for 2 channels
    output_left=$OutDir$output-channel_L.wav
    output_right=$OutDir$output-channel_R.wav

    # get the output directory, and create it if necessary
    outdir=$(dirname $output_left)
    mkdir -p $outdir

    # finally, convert the file
        # -G : guard against clipping
        # --ignore-length: do not use mp3 header to determine length
        # --norm=0 : apply peak normalization (0 dBFS)
        # -r 8000 : resample to 8 kHz
        # -b 16 convert to 16 bits

    # First the left channel
    sox -G --ignore-length $input --norm=0 -r $TARGET_SR -b 16 $output_left remix 1 |& tee -a $LogFile
    echo "Left channel saved as $output_left" |& tee -a $LogFile

    # Then the right channel
    sox -G --ignore-length $input --norm=0 -r $TARGET_SR -b 16 $output_right remix 2 |& tee -a $LogFile
    echo "Right channel saved as $output_right" |& tee -a $LogFile
    echo "" |& tee -a $LogFile

done

#!/bin/bash

#########################################################################################

if [ $# == 0 ]; then
    echo "Description: A simple script to recursively resample a bunch of files
          in a directory. Only certain file extensions (mp3, aac, flac, wav) are
          considered.

          It takes 2 command line options: InDir and OutDir.
          The destination (outdir) is relative to the current directory of where you 
          were when the script was run. 16-bit LPCM mono WAV files are written to outdir.
          Stereo files are converted to mono by averaging the channels.

          Example: resample.sh audio/ resampled/ 0

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

# Set the channel mode
ChannelMode=$3

# make sure the output directory exists (create it if not)
mkdir -p "$OutDir"

# Target sample rate
TARGET_SR=8000

# Target num channels
TARGET_NC=1

# Target bit depth
TARGET_BD=16

# start a log file
LogFile=$OutDir"/sox_log.txt"

# remove the log file if it already exists
rm -f $LogFile

# Convert each file with SoX, and write the converted file
# to the corresponding output dir, preserving the internal
# structure of the input dir
find $InDir -regextype posix-extended -type f -iregex '.*\.(mp3|wav|flac|aac)$' -print0 | while read -d $'\0' input
do

    echo "processing" $input |& tee -a $LogFile

    # the output path, without the InDir prefix
    output=${input#$InDir}
    # replace the original extension with .wav
    output=$OutDir/${output%.*}.wav

    # get the output directory, and create it if necessary
    outdir=$(dirname $output)
    mkdir -p "$outdir"

    # finally, convert the file
        # -G : guard against clipping
        # --ignore-length: do not use mp3 header to determine length
        # --norm=0 : apply peak normalization (0 dBFS)
        # -r 8000 : resample to 8 kHz
        # -b 16 convert to 16 bits
        # -c 1 mix the channels down to mono by averaging
    sox -G --ignore-length $input --norm=0 -r $TARGET_SR -b $TARGET_BD -c $TARGET_NC $output |& tee -a $LogFile

    echo "saved as $output" |& tee -a $LogFile

done

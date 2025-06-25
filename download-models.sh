#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Define URLs and Filenames ---
URL1="https://zenodo.org/records/15719945/files/nmfp-multiposcon.zip?download=1"
FILE1="nmfp-multiposcon.zip"

URL2="https://zenodo.org/records/15719945/files/nmfp-triplet.zip?download=1"
FILE2="nmfp-triplet.zip"

# --- 2. Define Target Directory ---
TARGET_DIR="logs/nmfp/fma-nmfp_deg/checkpoint/"

# --- 3. Download the files ---
echo "Downloading $FILE1..."
# Use wget to download the file and specify the output filename with -O
wget -O "$FILE1" "$URL1"
echo "Download of $FILE1 complete."

echo "---------------------------------"

echo "Downloading $FILE2..."
wget -O "$FILE2" "$URL2"
echo "Download of $FILE2 complete."

# --- 4. Create the target directory ---
echo "---------------------------------"
echo "Creating directory: $TARGET_DIR"
# Use -p to create parent directories as needed and avoid errors if it already exists
mkdir -p "$TARGET_DIR"
echo "Directory created."

# --- 5. Move the zip files to the target directory ---
echo "---------------------------------"
echo "Moving $FILE1 and $FILE2 to $TARGET_DIR"
mv "$FILE1" "$FILE2" "$TARGET_DIR"
echo "Files moved."

# --- 6. Unzip the files in the target directory ---
echo "---------------------------------"
echo "Changing directory to $TARGET_DIR"
cd "$TARGET_DIR"

echo "Unzipping $FILE1..."
unzip "$FILE1"
echo "$FILE1 unzipped."

echo "---------------------------------"

echo "Unzipping $FILE2..."
unzip "$FILE2"
echo "$FILE2 unzipped."

echo "---------------------------------"
echo "Script finished successfully."

# Optional: Return to the original directory
cd - > /dev/null

#!/bin/bash

# Define the URL and the file name
url="https://zenodo.org/records/15736620/files/neural-music-fp-dataset.tar.xz?download=1"
filename="neural-music-fp-dataset.tar.xz"

# Download the file using wget
wget -O "$filename" "$url"

# Unzip the tar.xz file
tar -xvf "$filename"

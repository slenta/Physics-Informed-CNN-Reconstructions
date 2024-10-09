#!/bin/bash

# Define the directory containing the files
data_dir="../data/masks/"

# Generate a comma-separated list of files
data_files=$(ls ${data_dir}en4_masks_1958-2023*.json | paste -sd "," -)

# Replace the placeholder in the configuration file with the comma-separated list
sed -i "s|--mask-names .*|--mask-names ${data_files}|" ../input/levante/train-cnn.txt

echo "Updated train-cnn.txt with data files: ${data_files}"

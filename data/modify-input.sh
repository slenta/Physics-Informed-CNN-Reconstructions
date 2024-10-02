#!/bin/bash

# Define the directory containing the files
data_dir="../data/test/"

# Generate a comma-separated list of files
data_files=$(ls ${data_dir}assimilation-test*.json | paste -sd "," -)

# Replace the placeholder in the configuration file with the comma-separated list
sed -i "s|--test-names .*|--test-names ${data_files}|" ../input/levante/train-cnn.txt

echo "Updated train-cnn.txt with data files: ${data_files}"

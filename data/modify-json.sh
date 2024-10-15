#!/bin/bash

# Define the directory containing the JSON files
json_dir="./test/"

# Define the old and new phrases
old_phrase="/assimilation_nc_files/"
new_phrase="/input-goratz/"

# Find all JSON files in the directory and replace the phrase
find "$json_dir" -name "*.json" | while read -r file; do
  sed -i "s#$old_phrase#$new_phrase#g" "$file"
  echo "Updated $file"
done

echo "All JSON files have been updated."

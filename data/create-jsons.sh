#!/bin/bash

# Base directory and file path template
base_dir="/work/uo1075/u301617/Master_Arbeit/data/input-goratz/input-masks"
file_template="en4_1958-2023_1744x872_GR15L40_"

# Loop to create 20 JSON files
for i in {1..20}
do
  # Generate the file path with the current number
  file_path="${base_dir}/level-${i}/${file_template}level-${i}.nc"
  
  # Create the JSON content
  json_content=$(cat <<EOF
[
  "${file_path}"
]
EOF
  )
  
  # Write the JSON content to a new file
  echo "${json_content}" > "./masks/en4_masks_1958-2023-level-${i}.json"
done

echo "Created 20 JSON files with varying file paths."

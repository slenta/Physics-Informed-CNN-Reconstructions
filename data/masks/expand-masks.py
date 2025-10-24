import xarray as xr
import json
import os
import pandas as pd
import numpy as np


# Function to duplicate the time dimension
def duplicate_time_dimension(input_file, output_file, num_times=40):
    # Open the input NetCDF file
    ds = xr.open_dataset(input_file, decode_times=False)
    print(input_file)

    # Repeat the first timestep across the time dimension
    repeated = [ds.copy(deep=True) for _ in range(num_times)]

    # Concatenate along time dimension
    duplicated_ds = xr.concat(repeated, dim="time")

    # Save the new dataset to a NetCDF file
    duplicated_ds.to_netcdf(output_file)
    print(
        f"Processed and saved: {output_file}, new shape: {len(duplicated_ds.time.values)}"
    )


# Function to process multiple NetCDF files listed in a JSON
def process_files_from_json(json_file, output_dir, num_times=40):
    # Open the JSON file and load the list of NetCDF file paths
    with open(json_file, "r") as f:
        netcdf_files = json.load(f)

    # Process each NetCDF file
    for input_file in netcdf_files:
        if os.path.exists(input_file):
            # Insert '_long' before the .nc extension
            base = os.path.splitext(os.path.basename(input_file))[0]
            output_file = f"{output_dir}/{base}_long.nc"
            duplicate_time_dimension(input_file, output_file, num_times)
        else:
            print(f"File not found: {input_file}")


# Example usage
lead_year = 3
json_file = f"/work/bk1318/k202208/crai/hindcast-pp/Physics-Informed-CNN-Reconstructions/data/masks/co2/test/co2-mpi-esm-lr-ly{lead_year}-acc-masks.json"  # Path to the JSON file containing the list of NetCDF files
num_times = 40  # Number of timesteps you want to create
output_dir = "/work/bk1318/k202208/crai/hindcast-pp/data/co2-flux/masks/MPI-ESM1-2-LR-co2flux-dcppA-hindcast-acc/long"

# Process all the files listed in the JSON
process_files_from_json(json_file, output_dir, num_times)

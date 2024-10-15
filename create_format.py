import xarray as xr
import json
import numpy as np

def create_format_info(nc_file):
    # Load the NetCDF file
    ds = xr.open_dataset(nc_file)
    ds = ds.squeeze("depth")

    # Extract dimensions and coordinates
    dimensions = list(ds.dims.keys())
    axes = list(ds.coords.keys())

    # Extract grid information
    grid = []
    step = []
    for dim in dimensions:
        coord_values = ds[dim].values
        grid.append([float(coord_values.min()), float(coord_values.max()), float(np.unique(np.diff(coord_values))[0])])
        step.append(float(np.unique(np.diff(coord_values))[0]))

    # Create the format information dictionary
    format_info = {
        "dimensions": dimensions,
        "axes": axes,
        "grid": grid,
        "step": step,
        "scale": [None, None]  # Placeholder for scale information
    }
    print(format_info)

nc_file = "../data/input-goratz/original_files/level-1/thetao_Omon_MPI-ESM-LR_asSEIKERAf_r1i8p4_195801-202010_anomalies_level-1.nc"
create_format_info(nc_file)

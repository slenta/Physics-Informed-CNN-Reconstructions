"""
Compute global Standardized Precipitation Evapotranspiration Index (SPEI)
from pre-calculated surplus (P - PET) using the Beguería et al. (2010) method
(log-logistic distribution) and the modern 'spei' Python package (Vonk, 2025).

References:
- Vicente-Serrano, Beguería & López-Moreno (2010), J. Climate 23(7):1696–1718.
- Vonk, M.A. (2025). "SPEI: A Python package for calculating and visualizing drought indices."

Requires:
    pip install numpy pandas xarray scipy spei tqdm

Usage:
    python spei_calc.py --input surplus_file.nc --output spei_output.nc --timescale 3
"""

# Set thread limits BEFORE importing numpy/scipy
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as sps
import spei as si
from IPython import embed
from tqdm import tqdm


def compute_spei_from_surplus(
    surplus: xr.DataArray,
    timescale: int = 3,
    var_names: list = ["CWB", "spei"],
    dist=sps.fisk,
) -> xr.DataArray:
    """
    Compute SPEI globally at each grid cell from pre-calculated surplus (P - PET).

    Parameters
    ----------
    surplus : xr.DataArray
        Monthly surplus (precipitation minus PET) [mm].
        Must have dims ('time', 'latitude', 'longitude').
    timescale : int, optional
        Accumulation timescale for SPEI (months). Default = 3.
    var_names : str, optional
        Output variable name in the resulting xarray. Default = 'SPEI'.
    dist : scipy.stats distribution, optional
        Distribution for SPEI standardization. Default = sps.fisk (log-logistic).

    Returns
    -------
    xr.DataArray
        Global SPEI with shape (time, latitude, longitude),
        standardized (mean=0, std=1).

    Notes
    -----
    - Standardization via log-logistic (Fisk) distribution.
    - Missing or constant data per grid cell are skipped gracefully.
    - NaN values converted to 0.
    """

    time = surplus["time"]
    lats = surplus["latitude"].values
    lons = surplus["longitude"].values
    n_time = len(time)

    # Initialize output with full time dimension
    spei_out = xr.DataArray(
        data=np.full((n_time, len(lats), len(lons)), np.nan, dtype=np.float32),
        coords={
            "time": time,
            "latitude": lats,
            "longitude": lons,
        },
        dims=["time", "latitude", "longitude"],
        name=var_names[1],
    )

    nlat, nlon = len(lats), len(lons)
    actual_output_length = None

    for i in tqdm(range(nlat), desc="Computing SPEI (per latitude)"):
        for j in range(nlon):
            surplus_grid = surplus[:, i, j].values

            # Skip if all zeros or all NaNs
            if np.all(surplus_grid == 0) or np.all(np.isnan(surplus_grid)):
                continue

            # Take out NaNs
            surplus_clean = np.nan_to_num(surplus_grid, nan=0.0)

            surplus_series = pd.Series(surplus_clean, index=pd.to_datetime(time.values))
            spei_series = si.spei(series=surplus_series, dist=dist, timescale=timescale)

            # Determine actual output length from first valid calculation
            if actual_output_length is None:
                actual_output_length = len(spei_series)

            # Place SPEI values at the end (aligned with input time)
            n_valid = len(spei_series)
            spei_out[-n_valid:, i, j] = spei_series.values

    # Slice to actual output length (remove initial NaN timesteps)
    if actual_output_length is not None:
        spei_out = spei_out.isel(time=slice(-actual_output_length, None))

    spei_out.attrs.update(
        {
            "description": f"{timescale}-month SPEI",
            "standardization": "log-logistic (Fisk)",
            "creator": "compute_spei_from_surplus",
            "timescale": timescale,
            "note": "NaN values converted to 0",
        }
    )

    return spei_out


if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    # Edit these variables directly when running the script

    input_file = "/work/bk1318/k202208/crai/hindcast-pp/data/spei/cwb/era5/era5-tamsat_cwb_remapped_invlat.nc"  # Path to input NetCDF file containing surplus (P - PET)
    output_file = "/work/bk1318/k202208/crai/hindcast-pp/data/spei/era5/era5-tamsat_spei-fisk_remapped_invlat.nc"  # Path to output NetCDF file for SPEI
    timescale = 3  # SPEI timescale in months
    var_names = ["CWB", "spei"]  # Variable name for output SPEI
    distribution = (
        sps.fisk
    )  # Distribution for SPEI standardization (e.g., sps.fisk, sps.gamma, sps.norm)

    # ===================================

    print(f"Computing SPEI with {timescale}-month timescale")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    # Load surplus data
    surplus = xr.open_dataset(input_file)[var_names[0]].squeeze()

    # Validate dimensions
    required_dims = {"time", "latitude", "longitude"}
    if not required_dims.issubset(set(surplus.dims)):
        raise ValueError(
            f"Input data must have dimensions {required_dims}, "
            f"but found {set(surplus.dims)}"
        )

    # Compute SPEI from surplus
    spei = compute_spei_from_surplus(
        surplus, timescale=timescale, var_names=var_names, dist=distribution
    )

    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    spei.to_netcdf(output_file)
    print(f"SPEI computation complete. Saved to {output_file}")

"""
Compute global climatic water balance (PET - Precipitation).

References:
- Vicente-Serrano, Beguería & López-Moreno (2010), J. Climate 23(7):1696–1718.
- Vonk, M.A. (2025). "SPEI: A Python package for calculating and visualizing drought indices."

Requires:
    pip install numpy pandas xarray scipy spei tqdm

Usage:
    python cwb_calc.py
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
from tqdm import tqdm
from pet import pet
from IPython import embed
import argparse


def compute_cwb_global(
    precip: xr.DataArray,
    temp: xr.DataArray,
    var_name: str = "CWB",
    temp_unit: str = "K",
) -> xr.Dataset:
    """
    Compute CWB globally at each grid cell for all ensemble members.

    Parameters
    ----------
    precip : xr.DataArray
        Monthly total precipitation [mm]. Must have dims ('time', 'member', 'latitude', 'longitude').
    temp : xr.DataArray
        Monthly mean temperature [°C]. Must have dims ('time', 'member', 'latitude', 'longitude').
    var_name : str, optional
        Output variable name in the resulting xarray. Default = 'CWB'.
    temp_unit : str, optional
        Temperature unit ('K' or 'C'). Default = 'K'.

    Returns
    -------
    xr.Dataset
        Dataset containing CWB, PET, and Precip with shape (time, member, latitude, longitude).

    Notes
    -----
    - PET computed via Hamon method.
    - CWB = Precipitation - PET
    - PET converted from daily to monthly by multiplying with days in month
    """

    # Align data
    precip, temp = xr.align(precip, temp)

    time = precip["time"]
    lats = precip["latitude"].values
    lons = precip["longitude"].values
    n_time = len(time)
    n_members = precip.shape[1]

    # Calculate days per month for each timestep
    time_pd = pd.to_datetime(time.values)
    days_in_month = np.array([pd.Period(t, freq="M").days_in_month for t in time_pd])

    # Initialize outputs with full dimensions
    cwb_out = xr.DataArray(
        data=np.full(
            (n_time, n_members, len(lats), len(lons)), np.nan, dtype=np.float32
        ),
        coords={
            "time": time,
            "member": np.arange(n_members),
            "latitude": lats,
            "longitude": lons,
        },
        dims=["time", "member", "latitude", "longitude"],
        name=var_name,
    )

    pet_out = cwb_out.copy(deep=True)
    pet_out.name = "PET"

    precip_out = cwb_out.copy(deep=True)
    precip_out.name = "Precip"

    nlat, nlon = len(lats), len(lons)

    for i in tqdm(range(nlat), desc="Computing CWB (per latitude)"):
        lat = lats[i]
        for j in range(nlon):
            for member in range(n_members):
                P = precip[:, member, i, j]
                T = temp[:, member, i, j]
                if temp_unit == "K":
                    T = T - 273.15  # Convert from K to °C

                # Skip if all zeros or all NaN
                if (np.all(P == 0) and np.all(T == 0)) or (
                    np.all(np.isnan(P)) and np.all(np.isnan(T))
                ):
                    continue

                # Compute daily PET
                PET_daily = pet(tmean=T, latitude=np.radians(lat), method="hamon")

                # Convert to monthly PET by multiplying with days in month
                PET = PET_daily * days_in_month
                cwb = P - PET

                cwb_out[:, member, i, j] = cwb
                pet_out[:, member, i, j] = PET
                precip_out[:, member, i, j] = P

    # Create Dataset with all variables
    ds = xr.Dataset({var_name: cwb_out, "PET": pet_out, "Precip": precip_out})

    ds.attrs.update(
        {
            "description": "Climatic Water Balance (Precipitation - Hamon PET)",
            "units": "mm",
            "creator": "compute_cwb_global",
            "ensemble_members": n_members,
            "note": "PET converted from daily to monthly totals",
        }
    )

    return ds


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compute CWB for all ensemble members at a specific lead year"
    )
    parser.add_argument(
        "--lead-year",
        type=int,
        default=0,
        help="Lead year for hindcast data (default: 0)",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="hc",
        help="Data type: 'hc' for hindcast, 'ref' for reference/ERA5 (default: 'hc')",
    )
    args = parser.parse_args()

    data_type = args.data_type
    lead_year = args.lead_year

    # Define paths
    data_path = "/work/bk1318/k202208/crai/hindcast-pp/data/spei/"
    # era5
    file_era5_precip = (
        data_path + "era5/precip/tamsat/tamsat_precip_monthly_1983-2024_era5grid.nc"
    )
    file_era5_tas = (
        data_path
        + "era5/tas/era5-monthly/tas_Amon_reanalysis_era5_r1i1p1_19400101-20241231_tamsatregion.nc"
    )

    # hindcast data
    hc_file_precip = (
        data_path + f"hindcasts/precip/dwd-hindcast_precip_{lead_year}-{lead_year}.nc"
    )
    hc_file_t2m = (
        data_path
        + f"hindcasts/t2m/full-values/dwd-hindcast_t2m_{lead_year}-{lead_year}.nc"
    )
    output_path = f"/work/bk1318/k202208/crai/hindcast-pp/data/spei/cwb/"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define output file
    hc_output_file = output_path + f"dwd-hindcast_cwb_ly{lead_year}.nc"
    ref_output_file = output_path + f"era5-tamsat_cwb_ly{lead_year}.nc"

    print(f"Computing CWB for all ensemble members (lead year {lead_year})")
    if data_type == "ref":
        print("Using reference/ERA5 data")
        output_file = ref_output_file
        file_precip = file_era5_precip
        file_t2m = file_era5_tas
    else:
        print("Using hindcast data")
        output_file = hc_output_file
        file_precip = hc_file_precip
        file_t2m = hc_file_t2m
    print(f"Output will be saved to: {output_file}")

    # Load data
    precip = xr.open_dataarray(file_precip)
    temp = xr.open_dataarray(file_t2m)

    # Compute CWB for all members
    cwb = compute_cwb_global(precip, temp, var_name="CWB")

    # Save output
    cwb.to_netcdf(output_file)
    print(f"CWB computation complete. Saved to {output_file}")

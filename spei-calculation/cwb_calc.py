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
    var_names: list = ["t2m", "precip", "CWB", "latitude", "longitude"],
    temp_unit: str = "K",
    member: bool = True,
) -> xr.Dataset:
    """
    Compute CWB globally at each grid cell for all ensemble members.

    Parameters
    ----------
    precip : xr.DataArray
        Monthly total precipitation [mm]. Must have dims ('time', 'member', 'latitude', 'longitude').
    temp : xr.DataArray
        Monthly mean temperature [°C]. Must have dims ('time', 'member', 'latitude', 'longitude').
    var_names : list, optional
        List of variable names in the order [temperature, precipitation, output, lat_name, lon_name].
        Default = ['t2m', 'precip', 'CWB', 'latitude', 'longitude'].
    temp_unit : str, optional
        Temperature unit ('K' or 'C'). Default = 'K'.
    member : bool, optional
        Whether data has a member dimension. Default = True.

    Returns
    -------
    xr.Dataset
        Dataset containing CWB, PET, and Precip with shape (time, member, latitude, longitude).

    Notes
    -----
    - PET computed via Hamon method.
    - CWB = Precipitation - PET
    - PET converted from daily to monthly by multiplying with days in month
    - Time coordinates normalized to first day of month for alignment
    """

    # Extract variables
    temp = temp[var_names[0]]
    precip = precip[var_names[1]]

    # Normalize time coordinates to the first day of each month
    temp_time = pd.to_datetime(temp["time"].values)
    precip_time = pd.to_datetime(precip["time"].values)

    # Convert to first day of month for both
    temp["time"] = pd.DatetimeIndex(
        [pd.Timestamp(t.year, t.month, 1) for t in temp_time]
    )
    precip["time"] = pd.DatetimeIndex(
        [pd.Timestamp(t.year, t.month, 1) for t in precip_time]
    )

    # Now align on the normalized time coordinate
    precip, temp = xr.align(precip, temp, join="inner")

    time = precip["time"]
    lats = precip[var_names[3]].values
    lons = precip[var_names[4]].values
    n_time = len(time)
    n_members = precip.shape[1] if member else 1

    # Calculate days per month for each timestep
    time_pd = pd.to_datetime(time.values)
    days_in_month = np.array([pd.Period(t, freq="M").days_in_month for t in time_pd])

    # Create properly formatted coordinate DataArrays with metadata
    lat_coord = xr.DataArray(
        lats,
        dims=["latitude"],
        attrs={
            "units": "degrees_north",
            "long_name": "latitude",
            "standard_name": "latitude",
            "axis": "Y",
        },
    )

    lon_coord = xr.DataArray(
        lons,
        dims=["longitude"],
        attrs={
            "units": "degrees_east",
            "long_name": "longitude",
            "standard_name": "longitude",
            "axis": "X",
        },
    )

    time_coord = xr.DataArray(
        time.values,
        dims=["time"],
        attrs={
            "long_name": "time",
            "standard_name": "time",
            "axis": "T",
        },
    )

    member_coord = xr.DataArray(
        np.arange(n_members) if member else [0],
        dims=["member"],
        attrs={
            "long_name": "ensemble member",
            "standard_name": "realization",
        },
    )

    # Initialize outputs with full dimensions and proper coordinates
    cwb_out = xr.DataArray(
        data=np.full(
            (n_time, n_members, len(lats), len(lons)),
            np.nan,
            dtype=np.float32,
        ),
        coords={
            "time": time_coord,
            "member": member_coord,
            "latitude": lat_coord,
            "longitude": lon_coord,
        },
        dims=["time", "member", "latitude", "longitude"],
        name=var_names[2],
        attrs={
            "units": "mm",
            "long_name": "Climatic Water Balance",
            "description": "Precipitation minus Potential Evapotranspiration",
        },
    )

    pet_out = cwb_out.copy(deep=True)
    pet_out.name = "PET"
    pet_out.attrs.update(
        {
            "long_name": "Potential Evapotranspiration",
            "description": "Monthly PET computed using Hamon method",
        }
    )

    precip_out = cwb_out.copy(deep=True)
    precip_out.name = "Precip"
    precip_out.attrs.update(
        {
            "long_name": "Precipitation",
            "description": "Monthly total precipitation",
        }
    )

    nlat, nlon = len(lats), len(lons)

    for i in tqdm(range(nlat), desc="Computing CWB (per latitude)"):
        lat = lats[i]
        for j in range(nlon):
            if member:
                for m in range(n_members):
                    P = precip[:, m, i, j]
                    T = temp[:, m, i, j]
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

                    cwb_out[:, m, i, j] = cwb
                    pet_out[:, m, i, j] = PET
                    precip_out[:, m, i, j] = P
            else:
                P = precip[:, i, j]
                T = temp[:, i, j]
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

                cwb_out[:, 0, i, j] = cwb
                pet_out[:, 0, i, j] = PET
                precip_out[:, 0, i, j] = P

    # Create Dataset with all variables
    ds = xr.Dataset({var_names[2]: cwb_out, "PET": pet_out, "Precip": precip_out})

    ds.attrs.update(
        {
            "title": "Climatic Water Balance (Precipitation - Hamon PET)",
            "description": "Monthly CWB, PET, and Precipitation data",
            "units": "mm",
            "creator": "compute_cwb_global",
            "ensemble_members": n_members,
            "note": "PET converted from daily to monthly totals",
            "Conventions": "CF-1.8",
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
    ref_output_file = output_path + f"era5-tamsat_cwb_ly.nc"

    print(
        f"Computing CWB for all ensemble members (lead year {lead_year}), data type: {data_type}"
    )
    if data_type == "ref":
        print("Using reference/ERA5 data")
        output_file = ref_output_file
        file_precip = file_era5_precip
        file_t2m = file_era5_tas
        var_names = ["tas", "rainfall_estimate_filled", "CWB", "lat", "lon"]
        member = False
    else:
        print("Using hindcast data")
        output_file = hc_output_file
        file_precip = hc_file_precip
        file_t2m = hc_file_t2m
        var_names = ["t2m", "precip", "CWB", "latitude", "longitude"]
        member = True
    print(f"Output will be saved to: {output_file}")

    # Load data
    precip = xr.open_dataset(file_precip)
    temp = xr.open_dataset(file_t2m)

    # Compute CWB for all members
    cwb = compute_cwb_global(precip, temp, var_names=var_names, member=member)
    # Save output
    cwb.to_netcdf(output_file)
    print(f"CWB computation complete. Saved to {output_file}")

"""
Compute global Standardized Precipitation Evapotranspiration Index (SPEI)
using the Beguería et al. (2010) method (Thornthwaite PET + log-logistic distribution)
and the modern 'spei' Python package (Vonk, 2025).

References:
- Vicente-Serrano, Beguería & López-Moreno (2010), J. Climate 23(7):1696–1718.
- Vonk, M.A. (2025). "SPEI: A Python package for calculating and visualizing drought indices."

Requires:
    pip install numpy pandas xarray scipy spei tqdm

Usage:
    python spei_calc.py --member 0
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
import argparse
from IPython import embed


def compute_spei_global(
    precip: xr.DataArray,
    temp: xr.DataArray,
    member: int = 0,
    timescale: int = 3,
    var_name: str = "SPEI",
) -> xr.DataArray:
    """
    Compute SPEI globally at each grid cell for a single ensemble member.

    Parameters
    ----------
    precip : xr.DataArray
        Monthly total precipitation [mm]. Must have dims ('time', 'member', 'latitude', 'longitude').
    temp : xr.DataArray
        Monthly mean temperature [°C]. Must have dims ('time', 'member', 'latitude', 'longitude').
    member : int, optional
        Ensemble member index to compute SPEI for. Default = 0.
    timescale : int, optional
        Accumulation timescale for SPEI (months). Default = 3.
    var_name : str, optional
        Output variable name in the resulting xarray. Default = 'SPEI'.

    Returns
    -------
    xr.DataArray
        Global SPEI with shape (time, latitude, longitude),
        standardized (mean=0, std=1).

    Notes
    -----
    - PET computed via Hamon method.
    - Standardization via log-logistic (Fisk) distribution.
    - Missing or constant data per grid cell are skipped gracefully.
    - NaN values converted to 0.
    """

    # Align data
    precip, temp = xr.align(precip, temp)

    # Select specific ensemble member
    precip = precip[:, member, :, :]
    temp = temp[:, member, :, :]

    time = precip["time"]
    lats = precip["latitude"].values
    lons = precip["longitude"].values
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
        name=var_name,
    )

    nlat, nlon = len(lats), len(lons)
    actual_output_length = None

    for i in tqdm(
        range(nlat), desc=f"Computing SPEI for member {member} (per latitude)"
    ):
        lat = lats[i]
        for j in range(nlon):
            P = precip[:, i, j]
            T = temp[:, i, j]

            # Skip if all zeros
            if np.all(P == 0) and np.all(T == 0):
                continue

            PET = pet(tmean=T, latitude=np.radians(lat), method="hamon")

            # Take out NaNs
            P = np.nan_to_num(P, nan=0.0)
            PET = np.nan_to_num(PET, nan=0.0)

            surplus = pd.Series(P - PET, index=pd.to_datetime(time.values))
            spei_series = si.spei(series=surplus, dist=sps.fisk, timescale=timescale)

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
            "description": f"{timescale}-month SPEI (Hamon PET)",
            "standardization": "log-logistic (Fisk)",
            "creator": "compute_spei_global",
            "ensemble_member": member,
            "timescale": timescale,
            "note": "NaN values converted to 0",
        }
    )

    return spei_out


lead_year = 0
data_path = "/work/bk1318/k202208/crai/hindcast-pp/data/spei/hindcasts/"
file_precip = data_path + f"precip/dwd-hindcast_precip_{lead_year}-{lead_year}.nc"
file_t2m = data_path + f"t2m/full-values/dwd-hindcast_t2m_{lead_year}-{lead_year}.nc"

output_path = (
    f"/work/bk1318/k202208/crai/hindcast-pp/data/spei/spei_begueria/{lead_year}/"
)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compute SPEI for a specific ensemble member"
    )
    parser.add_argument(
        "--member",
        type=int,
        default=0,
        help="Ensemble member index to compute SPEI for (default: 0)",
    )
    parser.add_argument(
        "--timescale",
        type=int,
        default=3,
        help="SPEI timescale in months (default: 3)",
    )
    args = parser.parse_args()

    # Define output file with member information
    output_file = (
        output_path + f"dwd-hindcast_spei_{args.timescale}m_member{args.member}.nc"
    )

    print(
        f"Computing SPEI for ensemble member {args.member} with {args.timescale}-month timescale"
    )
    print(f"Output will be saved to: {output_file}")

    # Load data
    precip = xr.open_dataarray(file_precip)
    temp = xr.open_dataarray(file_t2m)

    # Compute SPEI for specific member
    spei = compute_spei_global(
        precip, temp, member=args.member, timescale=args.timescale, var_name="SPEI"
    )

    # Save output
    spei.to_netcdf(output_file)
    print(f"SPEI computation complete. Saved to {output_file}")

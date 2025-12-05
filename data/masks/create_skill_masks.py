import xarray as xr
import numpy as np
import json
import os
from IPython import embed


def create_skill_masks(gt_file, hc_file, rmse_thresh, corr_thresh, mask_output_prefix):
    """
    Compute RMSE and correlation masks between two files with (time, lat, lon) variables.

    Args:
        gt_file (str): Path to ground truth NetCDF file.
        hc_file (str): Path to hindcast NetCDF file.
        rmse_thresh (float): RMSE threshold for mask.
        corr_thresh (float): Correlation threshold for mask.
        mask_output_prefix (str): Prefix for output mask files.
    """
    ds_hc = xr.open_dataset(hc_file)
    # Use hc_file's time to subset gt_file
    hc_time = ds_hc["time"]
    ds_gt = xr.open_dataset(gt_file)
    ds_gt = ds_gt.resample(time="1Y").mean()
    ds_gt = ds_gt.sel(time=slice("1961-01-01", "2019-12-31"))

    # Extract the variable of interest
    gt = ds_gt["tas"].values
    hc = ds_hc["tas"].values

    # Compute RMSE along time axis
    rmse = np.mean(np.sqrt((hc - gt) ** 2), axis=0)

    # Compute ACC (anomaly correlation coefficient) along time axis using numpy.corrcoef
    gt_anom = gt - np.mean(gt, axis=0)
    hc_anom = hc - np.mean(hc, axis=0)

    # Reshape to (time, lat*lon)
    nlat, nlon = gt.shape[1], gt.shape[2]
    gt_flat = gt_anom.reshape(gt.shape[0], nlat * nlon)
    hc_flat = hc_anom.reshape(hc.shape[0], nlat * nlon)

    # Compute ACC for each grid cell
    acc = np.array(
        [
            np.corrcoef(gt_flat[:, i], hc_flat[:, i])[0, 1]
            for i in range(gt_flat.shape[1])
        ]
    )

    # Reshape back to (lat, lon)
    acc = acc.reshape(nlat, nlon)

    # Create binary masks
    rmse_mask = (rmse < rmse_thresh).astype(np.uint8)
    acc_mask = (acc > corr_thresh).astype(np.uint8)

    # Convert masks to xarray DataArrays
    lat = ds_gt["lat"].values
    lon = ds_gt["lon"].values
    rmse_mask_xr = xr.DataArray(
        rmse_mask,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="rmse_mask",
    )
    acc_mask_xr = xr.DataArray(
        acc_mask, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="acc_mask"
    )

    # Get the full time dimension from hc_file
    hc_time = ds_hc["time"].values  # shape (ntime,)

    # Broadcast masks along time dimension
    rmse_mask_xr = xr.DataArray(
        np.broadcast_to(rmse_mask, (len(hc_time), nlat, nlon)),
        dims=("time", "lat", "lon"),
        coords={"time": hc_time, "lat": lat, "lon": lon},
        name="rmse_mask",
    )
    acc_mask_xr = xr.DataArray(
        np.broadcast_to(acc_mask, (len(hc_time), nlat, nlon)),
        dims=("time", "lat", "lon"),
        coords={"time": hc_time, "lat": lat, "lon": lon},
        name="acc_mask",
    )

    # Save masks as NetCDF
    rmse_mask_xr.name = "tas"
    acc_mask_xr.name = "tas"
    rmse_mask_xr.to_netcdf(f"{mask_output_prefix}_rmse_mask.nc")
    acc_mask_xr.to_netcdf(f"{mask_output_prefix}_acc_mask.nc")

    return rmse_mask_xr, acc_mask_xr


# Path to your JSON file and the ground truth file
ly = 1
json_path = f"/work/bk1318/k202208/crai/hindcast-pp/Physics-Informed-CNN-Reconstructions/data/test/tas/mpi-esm-ly{ly}-test.json"
gt_file = "/work/bk1318/k202208/crai/hindcast-pp/data/tas/era5-monthly/remapped/tas_Amon_reanalysis_era5_r1i1p1_19400101-20241231_remapped_anomaly.nc"
rmse_thresh = 3
corr_thresh = 0.3
output_dir = f"/work/bk1318/k202208/crai/hindcast-pp/data/tas/binary-masks/mpi_ly{ly}"

os.makedirs(output_dir, exist_ok=True)

with open(json_path, "r") as f:
    file_list = json.load(f)

for hc_file in file_list:
    # Extract a name for output prefix
    base = os.path.splitext(os.path.basename(hc_file))[0]
    mask_output_prefix = os.path.join(output_dir, base)
    create_skill_masks(gt_file, hc_file, rmse_thresh, corr_thresh, mask_output_prefix)

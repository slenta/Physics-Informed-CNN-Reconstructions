import xarray as xr
import numpy as np
import json
import os
import pandas as pd
from IPython import embed


def create_skill_masks(
    gt_file,
    hc_file,
    rmse_thresh,
    corr_thresh,
    mask_output_prefix,
    var,
    timemean=True,
    ly=1,
):
    """
    Compute RMSE and correlation masks between two files with (time, lat, lon) variables.

    Args:
        gt_file (str): Path to ground truth NetCDF file.
        hc_file (str): Path to hindcast NetCDF file.
        rmse_thresh (float): RMSE threshold for mask.
        corr_thresh (float): Correlation threshold for mask.
        mask_output_prefix (str): Prefix for output mask files.
        var (list): Variable names [hc, gt] to extract.
        timemean (bool): Whether to resample to annual means.
    """
    ds_hc = xr.open_dataset(hc_file)
    # Use hc_file's time to subset gt_file
    ds_gt = xr.open_dataset(gt_file)

    if timemean:
        ds_gt = ds_gt.resample(time="1Y").mean()
        ds_hc = ds_hc.resample(time="1Y").mean()

    # Shift forward by leadmonth due to dwd shift error
    if var[0] == "CWB":
        print(f"Shifting hindcast time by {ly - 1} months for CWB variable")
        shifted_time = pd.to_datetime(ds_hc["time"].values) + pd.DateOffset(
            months=(ly - 1)
        )
        ds_hc = ds_hc.assign_coords(time=shifted_time)

    hc_time = ds_hc["time"]
    gt_time = ds_gt["time"]

    # Get overlapping time range
    time_start = max(hc_time.values.min(), gt_time.values.min())
    time_end = min(hc_time.values.max(), gt_time.values.max())

    # Select common time period for both datasets
    ds_hc = ds_hc.sel(time=slice(time_start, time_end))
    ds_gt = ds_gt.sel(time=slice(time_start, time_end))

    print(f"Common time period: {time_start} to {time_end}")
    print(f"GT time length: {len(ds_gt.time)}, HC time length: {len(ds_hc.time)}")

    # Extract the variable of interest
    gt = ds_gt[var[1]].values
    hc = ds_hc[var[0]].values

    if len(hc.shape) == 4:
        # Assuming shape is (time, member, lat, lon), calculate ensemble mean
        hc = hc.mean(axis=1)
    if len(gt.shape) == 4:
        gt = gt.squeeze(axis=1)

    # Compute RMSE along time axis
    rmse = np.nanmean(np.sqrt((hc - gt) ** 2), axis=0)

    # Compute ACC (anomaly correlation coefficient) along time axis
    gt_anom = gt - np.nanmean(gt, axis=0)
    hc_anom = hc - np.nanmean(hc, axis=0)

    # Reshape to (time, lat*lon)
    nlat, nlon = gt.shape[1], gt.shape[2]
    gt_flat = gt_anom.reshape(gt.shape[0], nlat * nlon)
    hc_flat = hc_anom.reshape(hc.shape[0], nlat * nlon)

    # Compute ACC for each grid cell (NaN-aware)
    acc = np.full(gt_flat.shape[1], np.nan)
    for i in range(gt_flat.shape[1]):
        # Find valid (non-NaN) indices
        valid_mask = ~(np.isnan(gt_flat[:, i]) | np.isnan(hc_flat[:, i]))
        if valid_mask.sum() > 1:  # Need at least 2 points for correlation
            gt_valid = gt_flat[valid_mask, i]
            hc_valid = hc_flat[valid_mask, i]

            # Manual correlation calculation
            if np.std(gt_valid) > 0 and np.std(hc_valid) > 0:
                acc[i] = np.corrcoef(gt_valid, hc_valid)[0, 1]

    # Reshape back to (lat, lon)
    acc = acc.reshape(nlat, nlon)
    print(np.nanmean(acc), np.nanmean(rmse))

    # Create binary masks
    rmse_mask = (rmse < rmse_thresh).astype(np.uint8)
    acc_mask = (acc > corr_thresh).astype(np.uint8)

    # Convert masks to xarray DataArrays
    lat = ds_gt["latitude"].values
    lon = ds_gt["longitude"].values

    # Get the full time dimension from hc_file
    hc_time = ds_hc["time"].values  # shape (ntime,)

    # Create RMSE and ACC maps (continuous values)
    rmse_map_xr = xr.DataArray(
        rmse,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        name="rmse",
        attrs={
            "long_name": "Root Mean Square Error",
            "units": "mm",
            "description": f"RMSE between {var[0]} hindcast and ground truth",
        },
    )

    acc_map_xr = xr.DataArray(
        acc,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        name="acc",
        attrs={
            "long_name": "Anomaly Correlation Coefficient",
            "units": "1",
            "description": f"ACC between {var[0]} hindcast and ground truth",
        },
    )

    # Broadcast masks along time dimension
    rmse_mask_xr = xr.DataArray(
        np.broadcast_to(rmse_mask, (len(hc_time), nlat, nlon)),
        dims=("time", "latitude", "longitude"),
        coords={"time": hc_time, "latitude": lat, "longitude": lon},
        name="rmse_mask",
    )
    acc_mask_xr = xr.DataArray(
        np.broadcast_to(acc_mask, (len(hc_time), nlat, nlon)),
        dims=("time", "latitude", "longitude"),
        coords={"time": hc_time, "latitude": lat, "longitude": lon},
        name="acc_mask",
    )

    # Save masks as NetCDF
    rmse_mask_xr.name = var[0]
    acc_mask_xr.name = var[0]
    rmse_mask_xr.to_netcdf(f"{mask_output_prefix}_rmse_mask.nc")
    acc_mask_xr.to_netcdf(f"{mask_output_prefix}_acc_mask.nc")

    # Save RMSE and ACC maps (continuous values)
    rmse_map_xr.to_netcdf(f"{mask_output_prefix}_rmse_map.nc")
    acc_map_xr.to_netcdf(f"{mask_output_prefix}_acc_map.nc")

    return rmse_mask_xr, acc_mask_xr, rmse_map_xr, acc_map_xr


# Path to your JSON file and the ground truth file
# cwb config
# ly = 1
# json_path = f"/work/bk1318/k202208/crai/hindcast-pp/Physics-Informed-CNN-Reconstructions/data/test/cwb/dwd-hindcasts_lm{ly}-test.json"
# gt_file = (
#     "/work/bk1318/k202208/crai/hindcast-pp/data/spei/cwb/era5-tamsat_cwb_remapped.nc"
# )
# rmse_thresh = 3
# corr_thresh = 0.1
# output_dir = f"/work/bk1318/k202208/crai/hindcast-pp/data/spei/cwb/binary-masks/dwd-hindcasts_ly{ly}"
# var = "CWB"
# mean = False

gt_file = "/work/bk1318/k202208/crai/hindcast-pp/data/spei/cwb/era5-tamsat_cwb_remapped_invlat.nc"
rmse_thresh = 45
corr_thresh = 0.70
var = ["CWB", "CWB"]
mean = False

# Loop over multiple ly values
ly_values = [1, 2, 3, 4, 5, 6]

for ly in ly_values:
    print(f"\n{'='*60}")
    print(f"Processing ly = {ly}")
    print(f"{'='*60}\n")

    json_path = f"/work/bk1318/k202208/crai/hindcast-pp/Physics-Informed-CNN-Reconstructions/data/test/cwb/dwd-hindcasts_lm{ly}-test.json"
    output_dir = f"/work/bk1318/k202208/crai/hindcast-pp/data/spei/cwb/binary-masks/dwd-hindcasts_cwb_maxthres_lm{ly}"

    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        file_list = json.load(f)

    for hc_file in file_list:
        # Extract a name for output prefix
        base = os.path.splitext(os.path.basename(hc_file))[0]
        mask_output_prefix = os.path.join(output_dir, base)
        create_skill_masks(
            gt_file,
            hc_file,
            rmse_thresh,
            corr_thresh,
            mask_output_prefix,
            var=var,
            timemean=mean,
            ly=ly,
        )

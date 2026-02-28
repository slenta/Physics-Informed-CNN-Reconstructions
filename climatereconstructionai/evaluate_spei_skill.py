import os

from .utils.evaluation import (
    infill,
    get_batch_size,
    get_xr_dss,
    plot_ensemble_correlation_maps,
    plot_gridwise_correlation,
    plot_gridwise_rmse,
    plot_ensemble_rmse_maps,
    standardize_longitude,
    crps_ensemble_grid,
)
import xarray as xr
import numpy as np
from IPython import embed
from . import config as cfg
from .utils import visualization as vs
from .utils.evaluation import mae_per_member_grid, brier_skill_score_between_ensembles


def evaluate_spei_skill(arg_file=None, prog_func=None):

    cfg.set_evaluate_args(arg_file, prog_func)

    eval_path = [f"{cfg.evaluation_dirs[0]}/data/{name}" for name in cfg.eval_names]

    # Load reference data
    if cfg.data_types[0] == "tas":
        ref_data = xr.open_dataset(cfg.reference_data)
        ref_data = ref_data.resample(time="1Y").mean()
    else:
        ref_data = xr.open_dataset(cfg.reference_data, decode_times=False)

    n_ens = len(cfg.eval_names)
    corr_array, mae_array, masks, gt_ens, out_ens = [], [], [], [], []

    for i in range(n_ens):
        ds_out = xr.open_dataset(f"{eval_path[i]}_ly{cfg.lead_year}_output.nc")
        ds_gt = xr.open_dataset(f"{eval_path[i]}_ly{cfg.lead_year}_gt.nc")
        ds_mask = xr.open_dataset(f"{eval_path[i]}_ly{cfg.lead_year}_mask.nc")

        time = ds_gt.time
        first_year = time.min().dt.year.values
        last_year = time.max().dt.year.values

        if cfg.data_types[0] == "tas":
            ref = ref_data.sel(time=slice(f"{first_year}-01-01", f"{last_year}-12-31"))[
                f"{cfg.data_types[0]}"
            ].values
        else:
            ref = np.squeeze(ref_data[f"{cfg.data_types[0]}"].values)

        output = np.squeeze(ds_out[f"{cfg.data_types[0]}"].values)
        gt = np.squeeze(ds_gt[f"{cfg.data_types[0]}"].values)
        mask = ds_mask[f"{cfg.data_types[0]}"].values
        if cfg.data_types[0] == "CWB":
            lats = ref_data["latitude"].values
            lons = ref_data["longitude"].values
        else:
            lats = ref_data["lat"].values
            lons = ref_data["lon"].values
            output, _, _ = standardize_longitude(output, lons, lats)
            gt, _, _ = standardize_longitude(gt, lons, lats)
            ref, _, _ = standardize_longitude(ref, lons, lats)

        # Correlation
        gt_corr = plot_gridwise_correlation(
            gt,
            ref,
            lat=lats,
            lon=lons,
            title=f"GT Correlation Member {i+1} LY {cfg.lead_year}",
            save_path=cfg.evaluation_dirs[0],
            mask=mask,
            plot=False,
        )
        output_corr = plot_gridwise_correlation(
            output,
            ref,
            lat=lats,
            lon=lons,
            title=f"Output Correlation Member {i+1} LY {cfg.lead_year}",
            save_path=cfg.evaluation_dirs[0],
            mask=mask,
            plot=False,
        )
        diff_corr = output_corr - gt_corr

        # MAE (mean absolute error) per-member grids
        gt_mae = np.nanmean(np.abs(gt - ref), axis=0)
        output_mae = np.nanmean(np.abs(output - ref), axis=0)
        diff_mae = output_mae - gt_mae

        land_mask = np.isnan(gt_corr)
        output_corr = np.where(land_mask, np.nan, output_corr)
        gt_corr = np.where(land_mask, np.nan, gt_corr)
        diff_corr = np.where(land_mask, np.nan, diff_corr)

        output_mae = np.where(land_mask, np.nan, output_mae)
        gt_mae = np.where(land_mask, np.nan, gt_mae)
        diff_mae = np.where(land_mask, np.nan, diff_mae)

        corr_array.append(np.stack([gt_corr, output_corr, diff_corr, mask[0]], axis=0))
        mae_array.append(np.stack([gt_mae, output_mae, diff_mae, mask[0]], axis=0))
        gt_ens.append(gt)
        out_ens.append(output)
        masks.append(mask)

    gt_ens = np.where(land_mask, np.nan, np.array(gt_ens))
    out_ens = np.where(land_mask, np.nan, np.array(out_ens))
    masks = np.array(masks)
    corr_array = np.array(corr_array)
    mae_array = np.array(mae_array)

    # Ensemble means
    ensemble_mean_corr = np.nanmean(corr_array[:, :3, :, :], axis=0, keepdims=True)
    ensemble_mean_mae = np.nanmean(mae_array[:, :3, :, :], axis=0, keepdims=True)

    ensemble_min_mask = np.nanmin(corr_array[:, 3:4, :, :], axis=0, keepdims=True)

    ensemble_mean_corr = np.concatenate([ensemble_mean_corr, ensemble_min_mask], axis=1)
    ensemble_mean_mae = np.concatenate([ensemble_mean_mae, ensemble_min_mask], axis=1)

    corr_array = np.concatenate([corr_array, ensemble_mean_corr], axis=0)
    mae_array = np.concatenate([mae_array, ensemble_mean_mae], axis=0)

    # Plot correlation and MAE maps (reuse existing plotting helpers)
    plot_ensemble_correlation_maps(
        corr_array,
        lat=lats,
        lon=lons,
        save_path=cfg.evaluation_dirs[0],
        title=f"Ensemble Correlation Maps LY {cfg.lead_year}",
    )

    plot_ensemble_rmse_maps(
        mae_array,
        lat=lats,
        lon=lons,
        save_path=cfg.evaluation_dirs[0],
        title=f"Ensemble MAE Maps LY {cfg.lead_year}",
    )

    # Prepare ensemble arrays with shape (time, ensemble, lat, lon)
    # Current gt_ens and out_ens are (n_ens, time, lat, lon) -> transpose
    gt_ens_arr = np.transpose(gt_ens, (1, 0, 2, 3))
    out_ens_arr = np.transpose(out_ens, (1, 0, 2, 3))

    # Compute CRPS for ensembles vs reference
    crps_out = crps_ensemble_grid(out_ens_arr, ref)
    crps_gt = crps_ensemble_grid(gt_ens_arr, ref)

    # Average CRPS over time to produce grid maps
    crps_out_grid = np.nanmean(crps_out, axis=0)
    crps_gt_grid = np.nanmean(crps_gt, axis=0)
    crps_diff = crps_out_grid - crps_gt_grid

    # Compute Brier Skill Score comparing output ensemble to gt ensemble
    bss, bs_out, bs_gt = brier_skill_score_between_ensembles(
        out_ens_arr, gt_ens_arr, ref
    )

    # Save or plot CRPS/BSS maps using existing map plot (pack into expected shape)
    crps_array = np.stack([crps_gt_grid, crps_out_grid, crps_diff, masks[0]], axis=0)
    crps_array = np.expand_dims(crps_array, 0)  # shape (1,4,lat,lon)
    plot_ensemble_rmse_maps(
        crps_array,
        lat=lats,
        lon=lons,
        save_path=cfg.evaluation_dirs[0],
        title=f"Ensemble CRPS Maps LY {cfg.lead_year}",
    )

    bss_array = np.stack([bs_gt, bs_out, bss, masks[0]], axis=0)
    bss_array = np.expand_dims(bss_array, 0)
    plot_ensemble_rmse_maps(
        bss_array,
        lat=lats,
        lon=lons,
        save_path=cfg.evaluation_dirs[0],
        title=f"Ensemble BSS Maps LY {cfg.lead_year}",
    )


if __name__ == "__main__":
    evaluate_spei_skill()

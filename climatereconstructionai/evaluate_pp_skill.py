import os

from .utils.evaluation import (
    infill,
    get_batch_size,
    get_xr_dss,
    plot_ensemble_correlation_maps,
    plot_gridwise_correlation,
)
import xarray as xr
import numpy as np
from IPython import embed
from . import config as cfg
from .utils import visualization as vs


def evaluate_pp_skill(arg_file=None, prog_func=None):

    cfg.set_evaluate_args(arg_file, prog_func)

    # Define evaluation paths
    eval_path = [f"{cfg.evaluation_dirs[0]}/data/{name}" for name in cfg.eval_names]

    # Load reference data
    if cfg.data_types[0] == "tas":
        ref_data = xr.open_dataset(cfg.reference_data)
        ref_data = ref_data.resample(time="1Y").mean()
    else:
        ref_data = xr.open_dataset(cfg.reference_data, decode_times=False)

    n_ens = len(cfg.eval_names)
    corr_array, masks, gt_ens, out_ens = [], [], [], []

    for i in range(n_ens):
        # Load output and gt for this ensemble member
        ds_out = xr.open_dataset(f"{eval_path[i]}_ly{cfg.lead_year}_output.nc")
        ds_gt = xr.open_dataset(
            f"{eval_path[i]}_ly{cfg.lead_year}_gt.nc"
        )  # (time, lat, lon)
        ds_mask = xr.open_dataset(f"{eval_path[i]}_ly{cfg.lead_year}_mask.nc")

        time = ds_gt.time
        first_year = time.min().dt.year.values
        last_year = time.max().dt.year.values

        if cfg.data_types[0] == "tas":
            ref = ref_data.sel(time=slice(f"{first_year}-01-01", f"{last_year}-12-31"))[
                f"{cfg.data_types[0]}"
            ].values
        else:
            ref = ref_data[f"{cfg.data_types[0]}"].values
        output = ds_out[f"{cfg.data_types[0]}"].values
        gt = ds_gt[f"{cfg.data_types[0]}"].values
        mask = ds_mask[f"{cfg.data_types[0]}"].values

        # Calculate gridwise correlations
        gt_corr = plot_gridwise_correlation(
            gt,
            ref,
            lat=None,
            lon=None,
            title=f"GT Correlation Member {i+1} LY {cfg.lead_year}",
            save_path=cfg.evaluation_dirs[0],
            mask=mask,
            plot=False,
        )
        output_corr = plot_gridwise_correlation(
            output,
            ref,
            lat=None,
            lon=None,
            title=f"Output Correlation Member {i+1} LY {cfg.lead_year}",
            save_path=cfg.evaluation_dirs[0],
            mask=mask,
            plot=False,
        )
        diff_corr = output_corr - gt_corr
        land_mask = np.isnan(gt_corr)
        output_corr = np.where(land_mask, np.nan, output_corr)

        # Stack for this member
        corr_array.append(np.stack([gt_corr, output_corr, diff_corr, mask[0]], axis=0))
        gt_ens.append(gt)
        out_ens.append(output)
        masks.append(mask)

    gt_ens = np.array(gt_ens)  # shape: (n_ens, time, lat, lon)
    out_ens = np.array(out_ens)  # shape: (n_ens, time, lat, lon)
    masks = np.array(masks)  # shape: (n_ens, time, lat, lon)
    corr_array = np.array(corr_array)  # shape: (n_ens, 4, lat, lon)

    # Calculate ensemble mean for first 3 channels (correlations)
    ensemble_mean_corr = np.nanmean(
        corr_array[:, :3, :, :], axis=0, keepdims=True
    )  # shape: (1, 3, lat, lon)

    # Calculate ensemble min for mask channel (4th channel)
    ensemble_min_mask = np.nanmin(
        corr_array[:, 3:4, :, :], axis=0, keepdims=True
    )  # shape: (1, 1, lat, lon)

    # Concatenate along channel dimension
    ensemble_mean = np.concatenate(
        [ensemble_mean_corr, ensemble_min_mask], axis=1
    )  # shape: (1, 4, lat, lon)

    # Concatenate ensemble mean to the end
    corr_array = np.concatenate(
        [corr_array, ensemble_mean], axis=0
    )  # shape: (n_ens+1, 4, lat, lon)

    # Plot all ensemble correlation maps
    plot_ensemble_correlation_maps(
        corr_array,
        lat=None,
        lon=None,
        save_path=cfg.evaluation_dirs[0],
        title=f"Ensemble Correlation Maps LY {cfg.lead_year}",
    )

    vs.create_ensemble_timeseries(
        gt_ens,
        out_ens,
        ref,
        save_path=cfg.evaluation_dirs[0],
        title=f"Ensemble Timeseries LY {cfg.lead_year}",
    )

    vs.create_example_maps(
        gt_ens,
        out_ens,
        mask=masks,
        reference=ref,
        save_path=cfg.evaluation_dirs[0],
        title=f"Example Maps LY {cfg.lead_year}",
    )


if __name__ == "__main__":
    evaluate_pp_skill()

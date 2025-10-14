import os

from .model.net import CRAINet
from .utils.evaluation import (
    infill,
    get_batch_size,
    get_xr_dss,
    plot_ensemble_correlation_maps,
    plot_gridwise_correlation,
)
from .utils.io import load_ckpt, load_model
from torch.utils.data import DataLoader
from .utils.netcdfloader import NetCDFLoader, FiniteSampler
import xarray as xr
import numpy as np
from IPython import embed
from . import config as cfg


def store_encoding(ds):
    global encoding
    encoding = ds["time"].encoding
    return ds


def evaluate(arg_file=None, prog_func=None):
    cfg.set_evaluate_args(arg_file, prog_func)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    if not os.path.exists(f"{cfg.evaluation_dirs[0]}/images"):
        os.makedirs(f"{cfg.evaluation_dirs[0]}/images")
    if not os.path.exists(f"{cfg.evaluation_dirs[0]}/data"):
        os.makedirs(f"{cfg.evaluation_dirs[0]}/data")

    n_models = len(cfg.model_names)

    eval_path = [f"{cfg.evaluation_dirs[0]}/data/{name}" for name in cfg.eval_names]
    output_names = {}
    count = 0
    for i_model in range(n_models):

        ckpt_dict = load_ckpt(
            f"{cfg.model_dir}/{cfg.model_names[i_model]}.pth",
            cfg.device,
        )

        if cfg.use_train_stats:
            data_stats = ckpt_dict["train_stats"]
        else:
            data_stats = None

        dataset_val = NetCDFLoader(
            cfg.data_root_dir,
            cfg.data_names,
            cfg.mask_dir,
            cfg.mask_names,
            "infill",
            cfg.data_types,
            cfg.time_steps,
            data_stats,
        )

        xrdss = get_xr_dss(cfg.xrdss_paths, cfg.data_types)

        n_samples = len(dataset_val)
        print(n_samples)
        print(f"Shape of sample input: {dataset_val[0][0].shape}")

        if data_stats is None:
            if cfg.normalize_data:
                print("* Warning! Using mean and std from current data.")
                if cfg.n_target_data != 0:
                    print(
                        "* Warning! Mean and std from target data will be used to renormalize output."
                        " Mean and std from training data can be used with use_train_stats option."
                    )
            data_stats = {"mean": dataset_val.img_mean, "std": dataset_val.img_std}

        image_sizes = dataset_val.img_sizes
        if cfg.conv_factor is None:
            cfg.conv_factor = max(image_sizes[0])

        print(cfg.n_channel_steps)
        if len(image_sizes) - cfg.n_target_data < 1:
            model = CRAINet(
                img_size=image_sizes[0],
                enc_dec_layers=cfg.encoding_layers[0],
                pool_layers=cfg.pooling_layers[0],
                in_channels=cfg.n_channel_steps,
                out_channels=cfg.out_channels,
                fusion_img_size=image_sizes[1],
                fusion_enc_layers=cfg.encoding_layers[1],
                fusion_pool_layers=cfg.pooling_layers[1],
                fusion_in_channels=(len(image_sizes) - 1 - cfg.n_target_data)
                * cfg.n_channel_steps,
                bounds=dataset_val.bounds,
            ).to(cfg.device)
        else:
            model = CRAINet(
                img_size=image_sizes[0],
                enc_dec_layers=cfg.encoding_layers[0],
                pool_layers=cfg.pooling_layers[0],
                in_channels=cfg.n_channel_steps,
                out_channels=cfg.out_channels,
                bounds=dataset_val.bounds,
            ).to(cfg.device)

        for k in range(len(ckpt_dict["labels"])):
            count += 1
            label = ckpt_dict["labels"][k]
            load_model(ckpt_dict, model, label=label)
            model.eval()
            batch_size = get_batch_size(model.parameters(), n_samples, image_sizes)
            iterator_val = iter(
                DataLoader(
                    dataset_val,
                    batch_size=batch_size,
                    sampler=FiniteSampler(len(dataset_val)),
                    num_workers=0,
                )
            )
            infill(
                model,
                iterator_val,
                eval_path,
                output_names,
                data_stats,
                xrdss,
                count,
            )

    for name in output_names:
        if len(output_names[name]) == 1 and len(output_names[name][1]) == 1:
            os.rename(output_names[name][1][0], f"{name}.nc")
        else:
            if not cfg.split_outputs:
                dss = []
                for i_model in output_names[name]:
                    dss.append(
                        xr.open_mfdataset(
                            output_names[name][i_model],
                            preprocess=store_encoding,
                            autoclose=True,
                            combine="nested",
                            data_vars="minimal",
                            concat_dim="time",
                            chunks={},
                        )
                    )
                    dss[-1] = dss[-1].assign_coords({"member": i_model})

                if len(dss) == 1:
                    ds = dss[-1].drop("member")
                else:
                    ds = xr.concat(dss, dim="member")

                ds["time"].encoding = encoding
                ds["time"].encoding["original_shape"] = len(ds["time"])
                ds = ds.transpose("time", ...).reset_coords(drop=True)
                ds.to_netcdf(f"{name}.nc")

                for i_model in output_names[name]:
                    for output_name in output_names[name][i_model]:
                        os.remove(output_name)
            else:
                dss = []
                for i_model in output_names[name]:
                    dss.append(
                        xr.open_mfdataset(
                            output_names[name][i_model],
                            preprocess=store_encoding,
                            autoclose=True,
                            combine="nested",
                            data_vars="minimal",
                            concat_dim="time",
                            chunks={},
                        )
                    )
                    dss[-1] = dss[-1].assign_coords({"member": i_model})

                if len(dss) == 1:
                    ds = dss[-1].drop("member")
                else:
                    ds = xr.concat(dss, dim="member")

                ds["time"].encoding = encoding
                ds["time"].encoding["original_shape"] = len(ds["time"])
                ds = ds.transpose("time", ...).reset_coords(drop=True)
                ds.to_netcdf(f"{name}.nc")

                for i_model in output_names[name]:
                    for output_name in output_names[name][i_model]:
                        os.remove(output_name)

    if cfg.hindcast_eval == True:
        # Load reference data
        ref_data = xr.open_dataset(cfg.reference_data)
        ref_data = ref_data.resample(time="1Y").mean()
        n_ens = len(cfg.eval_names)
        corr_array = []

        for i in range(n_ens):
            # Load output and gt for this ensemble member
            ds_out = xr.open_dataset(f"{eval_path[i]}_ly{cfg.lead_year}_output.nc")
            ds_gt = xr.open_dataset(
                f"{eval_path[i]}_ly{cfg.lead_year}_gt.nc"
            )  # (time, lat, lon)

            time = ds_gt.time
            first_year = time.min().dt.year.values
            last_year = time.max().dt.year.values

            ref = ref_data.sel(
                time=slice(f"{first_year}-01-01", f"{last_year}-12-31")
            ).tas.values
            output = ds_out.tas.values
            gt = ds_gt.tas.values

            # Calculate gridwise correlations
            gt_corr = plot_gridwise_correlation(
                gt,
                ref,
                lat=None,
                lon=None,
                title=f"GT Correlation Member {i+1} LY {cfg.lead_year}",
                save_path=cfg.evaluation_dirs[0],
            )
            output_corr = plot_gridwise_correlation(
                output,
                ref,
                lat=None,
                lon=None,
                title=f"Output Correlation Member {i+1} LY {cfg.lead_year}",
                save_path=cfg.evaluation_dirs[0],
            )
            diff_corr = output_corr - gt_corr

            # Stack for this member
            corr_array.append(np.stack([gt_corr, output_corr, diff_corr], axis=0))

        corr_array = np.array(corr_array)  # shape: (n_ens, 3, lat, lon)

        # Plot all ensemble correlation maps
        plot_ensemble_correlation_maps(
            corr_array,
            lat=None,
            lon=None,
            save_path=cfg.evaluation_dirs[0],
            title=f"Ensemble Correlation Maps LY {cfg.lead_year}",
        )


if __name__ == "__main__":
    evaluate()

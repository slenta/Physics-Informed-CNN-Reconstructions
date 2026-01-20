import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter
import xarray as xr
from IPython import embed
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .netcdfchecker import reformat_dataset
from .netcdfloader import load_steadymask
from .normalizer import renormalize
from .plotdata import plot_data
from .. import config as cfg
from tqdm import tqdm

plt.rcParams.update({"font.size": 16})


def create_snapshot_image(model, dataset, filename):
    data_dict = {}
    data_dict["image"], data_dict["mask"], data_dict["gt"], index = zip(
        *[dataset[int(i)] for i in cfg.eval_timesteps]
    )

    for key in data_dict.keys():
        data_dict[key] = torch.stack(data_dict[key]).to(cfg.device)

    with torch.no_grad():
        data_dict["output"] = model(data_dict["image"], data_dict["mask"])

    data_dict["infilled"] = (
        data_dict["mask"] * data_dict["image"]
        + (1 - data_dict["mask"]) * data_dict["output"]
    )

    keys = list(data_dict.keys())
    for key in keys:
        data_dict[key] = data_dict[key].to(torch.device("cpu"))

    # set mask
    data_dict["mask"] = 1 - data_dict["mask"]
    data_dict["image"] = np.ma.masked_array(data_dict["image"], data_dict["mask"])
    data_dict["mask"] = np.ma.masked_array(data_dict["mask"], data_dict["mask"])

    n_rows = sum([data_dict[key].shape[2] for key in keys])
    n_cols = data_dict["image"].shape[0]

    # plot and save data
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows)
    )
    fig.patch.set_facecolor("black")

    for j in range(n_cols):
        axes[0, j].text(
            0.4, 1, index[j], size=24, transform=axes[0, j].transAxes, color="white"
        )

    k = 0
    for key in keys:
        for c in range(data_dict[key].shape[2]):

            if cfg.vlim is None:
                vmin = data_dict[key][:, :, c, :, :].min().item()
                vmax = data_dict[key][:, :, c, :, :].max().item()
            else:
                vmin = cfg.vlim[0]
                vmax = cfg.vlim[1]

            axes[k, 0].text(
                -0.8,
                0.5,
                key
                + " "
                + str(c)
                + "\n"
                + "{:.3e}".format(vmin)
                + "\n"
                + "{:.3e}".format(vmax),
                size=24,
                va="center",
                transform=axes[k, 0].transAxes,
                color="white",
            )

            for j in range(n_cols):
                axes[k, j].axis("off")
                axes[k, j].imshow(
                    np.squeeze(data_dict[key][j][cfg.recurrent_steps, c, :, :]),
                    vmin=vmin,
                    vmax=vmax,
                )

            k += 1

    plt.subplots_adjust(wspace=0.012, hspace=0.012)
    plt.savefig(filename + ".jpg", bbox_inches="tight", pad_inches=0)
    plt.clf()
    plt.close("all")


def get_xr_dss(xr_dss_paths, data_types):
    # get xrdss
    xr_dss = []
    for i, path in enumerate(xr_dss_paths):
        print("Loading xarray dataset from {:s}".format(path))
        ds = xr.load_dataset(path, decode_times=True)

        # Adjust the depth dimension for both ds and ds1
        # ds = ds.isel(depth=slice(0, cfg.out_channels))
        ds1 = ds.copy()

        # ds = ds.drop_vars(data_types[i])
        dims = ds1[data_types[i]].dims
        coords = {
            key: ds1[data_types[i]].coords[key]
            for key in ds1[data_types[i]].coords
            if key != "time"
        }
        ds1 = ds1.drop_vars(ds1.keys())
        ds1 = ds1.drop_dims("time")
        ds1 = ds1.drop_dims("depth") if "depth" in ds1.dims else ds1
        ds1 = ds1.drop_dims("height") if "height" in ds1.dims else ds1
        xr_dss.append([ds, ds1, dims, coords])
    return xr_dss


def get_batch_size(parameters, n_samples, image_sizes):
    if cfg.maxmem is None:
        partitions = cfg.partitions
    else:
        model_size = 0
        for parameter in parameters:
            model_size += sys.getsizeof(parameter.storage())
        model_size = 3.5 * n_samples * model_size / 1e6
        data_size = (
            4 * n_samples * np.sum([np.prod(size) for size in image_sizes]) * 5 / 1e6
        )
        partitions = int(np.ceil((model_size + data_size) / cfg.maxmem))

    if partitions > n_samples:
        partitions = n_samples

    if partitions != 1:
        print("The data will be split in {} partitions...".format(partitions))

    return int(np.ceil(n_samples / partitions))


def infill(model, dataset, eval_path, output_names, data_stats, xr_dss, i_model):
    if not os.path.exists(cfg.evaluation_dirs[0]):
        os.makedirs("{:s}".format(cfg.evaluation_dirs[0]))

    steady_mask = load_steadymask(
        cfg.mask_dir, cfg.steady_masks, cfg.data_types, cfg.device
    )

    data_dict = {"image": [], "mask": [], "gt": [], "output": [], "infilled": []}

    for split in tqdm(range(dataset.__len__())):
        # TODO: implement evaluation for multiple data paths
        data_dict["image"], data_dict["mask"], data_dict["gt"], index = next(dataset)

        if split == 0 and cfg.create_graph:
            writer = SummaryWriter(log_dir=cfg.log_dir)
            writer.add_graph(model, [data_dict["image"], data_dict["mask"]])
            writer.close()

        # get results from trained network
        with torch.no_grad():
            data_dict["output"] = model(
                data_dict["image"].to(cfg.device), data_dict["mask"].to(cfg.device)
            )

        # Choose time step, if lstm applied?
        for key in ("image", "mask", "gt", "output"):
            data_dict[key] = data_dict[key][:, cfg.recurrent_steps, :, :, :].to(
                torch.device("cpu")
            )

        # Question: If different data types/depth levels are used --> same channel everywhere?
        for key in ("image", "mask", "gt"):
            data_dict[key] = data_dict[key][:, cfg.gt_channels, :, :]

        if steady_mask is not None:
            for key in ("image", "gt", "output"):
                data_dict[key][:, :, steady_mask.type(torch.bool)] = np.nan

        data_dict["infilled"] = 1 - data_dict["mask"]
        data_dict["infilled"] *= data_dict["output"]
        data_dict["infilled"] += data_dict["mask"] * data_dict["image"]

        data_dict["image"] /= data_dict["mask"]

        create_outputs(
            data_dict,
            eval_path,
            output_names,
            data_stats,
            xr_dss,
            i_model,
            split,
            index,
        )

        if cfg.progress_fwd is not None:
            cfg.progress_fwd[0](
                "Infilling...",
                int(
                    cfg.progress_fwd[2]
                    * (cfg.progress_fwd[1] + (split + 1) / dataset.__len__())
                ),
            )

    return output_names


def create_outputs(
    data_dict,
    eval_path,
    output_names,
    data_stats,
    xr_dss,
    i_model,
    split,
    index,
    ds_index=0,
):

    m_label = "." + str(i_model)
    suffix = m_label + "-" + str(split + 1)

    if cfg.n_target_data == 0:
        cnames = ["gt", "mask", "image", "output", "infilled"]
        pnames = ["image", "infilled"]
    else:
        cnames = ["gt", "output"]
        pnames = ["gt", "output"]

    split_data = []
    for cname in cnames:

        if cfg.normalize_data and cname != "mask":
            for k in range(data_dict[cname].shape[1]):
                data_dict[cname][:, k, :, :] = renormalize(
                    data_dict[cname][:, k, :, :],
                    data_stats["mean"][k],
                    data_stats["std"][k],
                )

        # Split data_dict[cname] along axis 0
        split_arrays = np.array_split(
            data_dict[cname].squeeze().to(torch.device("cpu")).detach().numpy(),
            len(eval_path),
            axis=0,
        )
        split_data.append(split_arrays)

    for j in range(len(eval_path)):

        if len(cfg.data_types) > 1:
            i_data = -cfg.n_target_data + j
        else:
            i_data = 0
        data_type = cfg.data_types[i_data]

        for c, cname in enumerate(cnames):
            rootname = f"{eval_path[j]}_ly{cfg.lead_year}_{cname}"
            if rootname not in output_names:
                output_names[rootname] = {}

            if i_model not in output_names[rootname]:
                output_names[rootname][i_model] = []

            output_names[rootname][i_model] += [rootname + suffix + ".nc"]

            ds = xr_dss[i_data][1].copy()

            ds[data_type] = xr.DataArray(
                split_data[c][j],
                dims=xr_dss[i_data][2],
                coords=xr_dss[i_data][3],
            )
            ds["time"] = xr_dss[i_data][0]["time"]

            ds = reformat_dataset(
                xr_dss[i_data][0], ds, data_type, format_name=cfg.eval_format
            )
            # for var in xr_dss[i_data][0].keys():
            #     if "time" in xr_dss[i_data][0][var].dims:
            #         ds[var] = xr_dss[i_data][0][var].isel(time=index)
            #     else:
            #         ds[var] = xr_dss[i_data][0][var]

            ds.attrs["history"] = (
                "Infilled using CRAI (Climate Reconstruction AI: "
                "https://github.com/FREVA-CLINT/climatereconstructionAI)\n"
                + ds.attrs["history"]
            )
            ds.to_netcdf(output_names[rootname][i_model][-1])

        for time_step in cfg.plot_results:
            if time_step in index:
                output_name = "{}_{}{}_{}.png".format(
                    eval_path[j], "combined", m_label, time_step
                )
                plot_data(
                    xr_dss[i_data][1].coords,
                    [
                        data_dict[p][time_step - index[0], j, :, :].squeeze()
                        for p in pnames
                    ],
                    ["Original", "Reconstructed"],
                    output_name,
                    data_type,
                    str(xr_dss[i_data][0]["time"][time_step].values),
                    *cfg.dataset_format["scale"],
                )


def standardize_longitude(data, lons, lats=None):
    """
    Standardize longitude to -180-180 range and latitude to -90-90 range, reordering data accordingly.

    Args:
        data: np.ndarray with longitude as last dimension and latitude as second-to-last
        lons: 1D array of longitude values
        lats: 1D array of latitude values (optional)

    Returns:
        data_reordered: data with longitudes reordered to -180-180 and latitudes to -90-90
        lons_reordered: longitude array in -180 - 180 range
        lats_reordered: latitude array in -90-90 range (if lats provided)
    """
    # Convert lons to -180-180 range
    lons_180 = np.where(lons > 180, lons - 360, lons)

    # Get sorting indices for longitude
    lon_sort_idx = np.argsort(lons_180)

    # Reorder lons
    lons_reordered = lons_180[lon_sort_idx]

    # Reorder data along longitude axis (last axis)
    data_reordered = np.take(data, lon_sort_idx, axis=-1)

    # Handle latitude reordering if provided
    if lats is not None:
        # Convert lats to -90 to 90 range
        lats_90 = np.where(lats > 90, lats - 180, lats)

        # Get sorting indices for latitude
        lat_sort_idx = np.argsort(lats_90)

        # Reorder lats
        lats_reordered = lats_90[lat_sort_idx]

        # Reorder data along latitude axis (second-to-last axis)
        data_reordered = np.take(data_reordered, lat_sort_idx, axis=-2)

        return data_reordered, lons_reordered, lats_reordered

    return data_reordered, lons_reordered


def plot_gridwise_correlation(
    tensor1,
    tensor2,
    save_path,
    mask=None,
    lat=None,
    lon=None,
    title="Gridwise Correlation",
    plot=True,
):
    """
    Calculate and plot gridwise correlation between two tensors of shape (time, lat, lon).

    Args:
        tensor1: np.ndarray or torch.Tensor, shape (time, lat, lon)
        tensor2: np.ndarray or torch.Tensor, shape (time, lat, lon)
        mask: np.ndarray or torch.Tensor, shape (lat, lon), binary mask (optional)
        lat: 1D array of latitude values (optional, for axis labeling)
        lon: 1D array of longitude values (optional, for axis labeling)
        title: Title for the plot
        save_path: If provided, saves the figure to this path
    """
    # Convert to numpy if torch tensor
    if hasattr(tensor1, "detach"):
        tensor1 = tensor1.detach().cpu().numpy()
    if hasattr(tensor2, "detach"):
        tensor2 = tensor2.detach().cpu().numpy()
    if mask is not None and hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    time, nlat, nlon = tensor1.shape

    # Reshape to (time, lat*lon)
    t1_flat = np.nan_to_num(tensor1.reshape(time, nlat * nlon), 0)
    t2_flat = np.nan_to_num(tensor2.reshape(time, nlat * nlon), 0)

    # Calculate correlation for each grid cell
    corr = np.array(
        [np.corrcoef(t1_flat[:, i], t2_flat[:, i])[0, 1] for i in range(nlat * nlon)]
    )
    corr_grid = corr.reshape(nlat, nlon)

    # Plot
    if plot == True:
        plt.figure(figsize=(8, 6))
        im = plt.imshow(corr_grid, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)

        # Add mask overlay with small dots
        if mask is not None:
            y_coords = np.where(mask == 1)[1]
            x_coords = np.where(mask == 1)[2]
            plt.scatter(
                x_coords,
                y_coords,
                s=0.003,
                c="black",
                alpha=0.8,
                marker="o",
                linewidths=0.1,
            )

        plt.colorbar(im, label="Correlation")
        plt.title(title)
        plt.xlabel("Longitude" if lon is not None else "Grid X")
        plt.ylabel("Latitude" if lat is not None else "Grid Y")
        if lat is not None and lon is not None:
            plt.xticks(np.arange(len(lon)), np.round(lon, 2), rotation=90)
            plt.yticks(np.arange(len(lat)), np.round(lat, 2))
        plt.tight_layout()
        plt.savefig(f"{save_path}/images/{title}.png", bbox_inches="tight")
        plt.close()

    return corr_grid


def plot_gridwise_rmse(
    tensor1,
    tensor2,
    save_path,
    mask=None,
    lat=None,
    lon=None,
    title="Gridwise RMSE",
    plot=True,
):
    """
    Calculate and plot gridwise RMSE between two tensors of shape (time, lat, lon).

    Args:
        tensor1: np.ndarray or torch.Tensor, shape (time, lat, lon)
        tensor2: np.ndarray or torch.Tensor, shape (time, lat, lon)
        mask: np.ndarray or torch.Tensor, shape (lat, lon), binary mask (optional)
        lat: 1D array of latitude values (optional, for axis labeling)
        lon: 1D array of longitude values (optional, for axis labeling)
        title: Title for the plot
        save_path: If provided, saves the figure to this path
        plot: If True, creates and saves the plot

    Returns:
        rmse_grid: np.ndarray, shape (lat, lon) - RMSE values for each grid cell
    """
    # Convert to numpy if torch tensor
    if hasattr(tensor1, "detach"):
        tensor1 = tensor1.detach().cpu().numpy()
    if hasattr(tensor2, "detach"):
        tensor2 = tensor2.detach().cpu().numpy()
    if mask is not None and hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    time, nlat, nlon = tensor1.shape

    # Calculate squared differences
    squared_diff = (tensor1 - tensor2) ** 2

    # Calculate RMSE for each grid cell (mean over time axis)
    rmse_grid = np.sqrt(np.nanmean(squared_diff, axis=0))  # shape: (nlat, nlon)

    # Plot
    if plot == True:
        plt.figure(figsize=(8, 6))
        im = plt.imshow(rmse_grid, origin="lower", cmap="YlOrRd")

        # Add mask overlay with small dots
        if mask is not None:
            y_coords, x_coords = np.where(mask == 1)
            plt.scatter(
                x_coords,
                y_coords,
                s=0.003,
                c="black",
                alpha=0.8,
                marker="o",
                linewidths=0.1,
            )

        plt.colorbar(im, label="RMSE")
        plt.title(title)
        plt.xlabel("Longitude" if lon is not None else "Grid X")
        plt.ylabel("Latitude" if lat is not None else "Grid Y")
        if lat is not None and lon is not None:
            plt.xticks(np.arange(len(lon)), np.round(lon, 2), rotation=90)
            plt.yticks(np.arange(len(lat)), np.round(lat, 2))
        plt.tight_layout()
        plt.savefig(f"{save_path}/images/{title}.png", bbox_inches="tight", dpi=300)
        plt.close()

    return rmse_grid


def plot_ensemble_correlation_maps(
    corr_array, lat=None, lon=None, save_path=None, title="Ensemble Correlation Maps"
):
    """
    Plots an array of correlation maps with shape (n_ens_members, 3 or 4, nlat, nlon).
    The 3 maps per member are: gt_correlation, output_correlation, differences.
    If 4 maps provided, the 4th is used as a binary mask for stippling.

    Args:
        corr_array: np.ndarray, shape (n_ens_members, 3 or 4, nlat, nlon)
        lat: 1D array of latitude values (optional)
        lon: 1D array of longitude values (optional)
        save_path: Directory to save the figure (optional)
        title: Title for the figure
    """
    n_ens, n_maps, nlat, nlon = corr_array.shape

    # Keep only first 3 members and last member
    indices = list(range(3)) + [-1]
    corr_array = corr_array[indices]
    n_ens = len(indices)

    # Extract mask if n_maps == 4
    mask = None
    if n_maps == 4:
        mask = corr_array[:, 3, :, :]  # shape: (n_ens, nlat, nlon)
        corr_array = corr_array[:, :3, :, :]  # Only plot first 3 maps
        n_maps = 3

    map_titles = ["GT Correlation", "Output Correlation", "Diff: Out - GT"]

    # Create figure
    fig = plt.figure(figsize=(6 * n_maps, 3 * n_ens))
    fig.suptitle(title, fontsize=18)

    for i in range(n_ens):
        for j in range(n_maps):
            # Create subplot with cartopy projection
            ax = fig.add_subplot(
                n_ens, n_maps, i * n_maps + j + 1, projection=ccrs.PlateCarree()
            )

            im = ax.imshow(
                corr_array[i, j],
                origin="lower",
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                extent=[min(lon), max(lon), min(lat), max(lat)],
                transform=ccrs.PlateCarree(),
            )

            # Add coastlines
            ax.coastlines(linewidth=0.5, color="black")
            ax.set_extent(
                [min(lon), max(lon), min(lat), max(lat)], crs=ccrs.PlateCarree()
            )

            # Add mask overlay with small dots (use ensemble-specific mask)
            if mask is not None:
                y_coords, x_coords = np.where(mask[i] == 1)
                ax.scatter(
                    x_coords * 360 / nlon + min(lon),
                    y_coords * 180 / nlat + min(lat),
                    s=0.1,
                    c="black",
                    alpha=0.5,
                    marker=".",
                    transform=ccrs.PlateCarree(),
                )

            # Set title - use "Ensemble Mean" for last member
            member_label = "Ensemble Mean" if i == n_ens - 1 else f"Member {i+1}"
            ax.set_title(f"{member_label}: {map_titles[j]}")

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(
        f"{save_path}/images/{title.replace(' ', '_')}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_ensemble_rmse_maps(
    rmse_array, lat=None, lon=None, save_path=None, title="Ensemble RMSE Maps"
):
    """
    Plots an array of RMSE maps with shape (n_ens_members, 3 or 4, nlat, nlon).
    The 3 maps per member are: gt_rmse, output_rmse, differences.
    If 4 maps provided, the 4th is used as a binary mask for stippling.

    Args:
        rmse_array: np.ndarray, shape (n_ens_members, 3 or 4, nlat, nlon)
        lat: 1D array of latitude values (optional)
        lon: 1D array of longitude values (optional)
        save_path: Directory to save the figure (optional)
        title: Title for the figure
    """
    n_ens, n_maps, nlat, nlon = rmse_array.shape

    # Keep only first 3 members and last member
    indices = list(range(3)) + [-1]
    rmse_array = rmse_array[indices]
    n_ens = len(indices)

    # Extract mask if n_maps == 4
    mask = None
    if n_maps == 4:
        mask = rmse_array[:, 3, :, :]  # shape: (n_ens, nlat, nlon)
        rmse_array = rmse_array[:, :3, :, :]  # Only plot first 3 maps
        n_maps = 3

    map_titles = ["GT RMSE", "Output RMSE", "RMSE Diff: Out - GT"]

    # Create figure
    fig = plt.figure(figsize=(6 * n_maps, 3 * n_ens))
    fig.suptitle(title, fontsize=18)

    # Compute global vmin/vmax for consistent color scale
    vmin = np.nanmin(rmse_array[-1, :2])
    vmax = np.nanmax(rmse_array[-1, :2])
    vmin_diff = np.nanmin(rmse_array[-1, 2])
    vmax_diff = np.nanmax(rmse_array[-1, 2])

    for i in range(n_ens):
        for j in range(n_maps):
            # Create subplot with cartopy projection
            ax = fig.add_subplot(
                n_ens, n_maps, i * n_maps + j + 1, projection=ccrs.PlateCarree()
            )

            if j != 2:
                im = ax.imshow(
                    rmse_array[i, j],
                    origin="lower",
                    cmap="YlOrRd",
                    vmin=vmin,
                    vmax=vmax,
                    extent=[min(lon), max(lon), min(lat), max(lat)],
                    transform=ccrs.PlateCarree(),
                )
            elif j == 2:
                im = ax.imshow(
                    rmse_array[i, j],
                    origin="lower",
                    cmap="coolwarm",
                    vmin=vmin_diff,
                    vmax=vmax_diff,
                    extent=[min(lon), max(lon), min(lat), max(lat)],
                    transform=ccrs.PlateCarree(),
                )

            # Add coastlines
            ax.coastlines(linewidth=0.5, color="black")
            ax.set_extent(
                [min(lon), max(lon), min(lat), max(lat)], crs=ccrs.PlateCarree()
            )

            # Add mask overlay with small dots (use ensemble-specific mask)
            if mask is not None:
                y_coords, x_coords = np.where(mask[i] == 1)
                ax.scatter(
                    x_coords * 360 / nlon + min(lon),
                    y_coords * 180 / nlat + min(lat),
                    s=0.1,
                    c="black",
                    alpha=0.5,
                    marker=".",
                    transform=ccrs.PlateCarree(),
                )

            # Set title - use "Ensemble Mean" for last member
            member_label = "Ensemble Mean" if i == n_ens - 1 else f"Member {i+1}"
            ax.set_title(f"{member_label}: {map_titles[j]}")

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RMSE")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(
        f"{save_path}/images/{title.replace(' ', '_')}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)

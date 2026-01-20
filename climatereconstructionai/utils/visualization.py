import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import embed
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def calculate_distributions(
    mask, steady_mask, output, gt, domain="valid", num_samples=1000
):
    if steady_mask is not None:
        mask += steady_mask
        mask[mask < 0] = 0
        mask[mask > 1] = 1
        assert (
            (mask == 0) | (mask == 1)
        ).all(), "Not all values in mask are zeros or ones!"

    value_list_pred = []
    value_list_target = []
    for ch in range(output.shape[2]):
        mask_ch = mask[:, :, ch, :, :]
        gt_ch = gt[:, :, ch, :, :]
        output_ch = output[:, :, ch, :, :]

        if domain == "valid":
            pred = output_ch[mask_ch == 1]
            target = gt_ch[mask_ch == 1]
        elif domain == "hole":
            pred = output_ch[mask == 0]
            target = gt_ch[mask == 0]
        elif domain == "comp_infill":
            pred = mask_ch * gt_ch + (1 - mask_ch) * output_ch
            target = gt_ch
        pred = pred.flatten()
        target = target.flatten()

        sample_indices = torch.randint(len(pred), (num_samples,))
        value_list_pred.append(pred[sample_indices])
        value_list_target.append(target[sample_indices])

    return value_list_pred, value_list_target


def calculate_error_distributions(
    mask, steady_mask, output, gt, operation="AE", domain="valid", num_samples=1000
):
    preds, targets = calculate_distributions(
        mask, steady_mask, output, gt, domain=domain, num_samples=num_samples
    )

    if steady_mask is not None:
        mask += steady_mask
        mask[mask < 0] = 0
        mask[mask > 1] = 1
        assert (
            (mask == 0) | (mask == 1)
        ).all(), "Not all values in mask are zeros or ones!"

    value_list = []
    for ch in range(len(preds)):
        pred = preds[ch]
        target = targets[ch]

        if operation == "AE":
            values = torch.sqrt((pred - target) ** 2)
        elif operation == "E":
            values = pred - target
        elif operation == "RAE":
            values = (pred - target).abs() / (target + 1e-9)
        elif operation == "RE":
            values = (pred - target) / (target + 1e-9)

        values = values.flatten()
        sample_indices = torch.randint(len(values), (num_samples,))
        value_list.append(values[sample_indices])
    return value_list


def create_error_dist_plot(
    mask, steady_mask, output, gt, operation="E", domain="valid", num_samples=1000
):
    preds, targets = calculate_distributions(
        mask, steady_mask, output, gt, domain=domain, num_samples=num_samples
    )

    fig, axs = plt.subplots(1, len(preds), squeeze=False)

    for ch in range(len(preds)):

        pred = preds[ch].cpu()
        target = targets[ch].cpu()

        if operation == "AE":
            errors_ch = np.sqrt((pred - target) ** 2)
        elif operation == "E":
            errors_ch = pred - target
        elif operation == "RAE":
            errors_ch = (pred - target).abs() / (target + 1e-9)
        elif operation == "RE":
            errors_ch = (pred - target) / (target + 1e-9)

        m = (errors_ch).mean()
        s = (errors_ch).std()
        xlims = [
            target.min() - 0.5 * (target).diff().abs().mean(),
            target.max() + 0.5 * (target).diff().abs().mean(),
        ]

        axs[0, ch].hlines(
            [m, m - s, m + s],
            xlims[0],
            xlims[1],
            colors=["grey", "red", "red"],
            linestyles="dashed",
        )
        axs[0, ch].scatter(target, errors_ch, color="black")
        axs[0, ch].grid()
        axs[0, ch].set_xlabel("target values")
        axs[0, ch].set_ylabel("errors")
        axs[0, ch].set_xlim(xlims)
    return fig


def create_correlation_plot(
    mask, steady_mask, output, gt, domain="valid", num_samples=1000
):
    preds, targets = calculate_distributions(
        mask, steady_mask, output, gt, domain=domain, num_samples=num_samples
    )

    fig, axs = plt.subplots(1, len(preds), squeeze=False)

    for ch in range(len(preds)):
        target_data = targets[ch]
        pred_data = preds[ch]
        R = torch.corrcoef(torch.vstack((target_data, pred_data)))[0, 1]

        axs[0, ch].scatter(target_data.cpu(), pred_data.cpu(), color="red", alpha=0.5)
        axs[0, ch].plot(target_data.cpu(), target_data.cpu(), color="black")
        axs[0, ch].grid()
        axs[0, ch].set_xlabel("target values")
        axs[0, ch].set_ylabel("predicted values")
        axs[0, ch].set_title(f"R = {R:.4}")

    return fig


def create_error_map(
    mask, steady_mask, output, gt, num_samples=3, operation="AE", domain="valid"
):
    if steady_mask is not None:
        mask += steady_mask
        mask[mask < 0] = 0
        mask[mask > 1] = 1
        assert (
            (mask == 0) | (mask == 1)
        ).all(), "Not all values in mask are zeros or ones!"

    num_channels = output.shape[2]
    samples = torch.randint(output.shape[0], (num_samples,))

    fig, axs = plt.subplots(
        num_channels,
        num_samples,
        squeeze=False,
        figsize=(num_samples * 7, num_channels * 7),
    )

    for ch in range(output.shape[2]):
        gt_ch = gt[:, :, ch, :, :]
        output_ch = output[:, :, ch, :, :]

        for sample_num in range(num_samples):

            target = np.squeeze(gt_ch[samples[sample_num]].squeeze())
            pred = np.squeeze(output_ch[samples[sample_num]].squeeze())

            if operation == "AE":
                values = (pred - target).abs()
                cm = "cividis"
            elif operation == "E":
                values = pred - target
                cm = "coolwarm"
            elif operation == "RAE":
                values = (pred - target).abs() / (target.abs() + 1e-9)
                cm = "cividis"
            elif operation == "RE":
                values = (pred - target) / (target + 1e-9)
                cm = "coolwarm"

            vmin, vmax = torch.quantile(
                values, torch.tensor([0.05, 0.95], device=values.device)
            )
            cp = axs[ch, sample_num].matshow(
                values.cpu(), cmap=cm, vmin=vmin, vmax=vmax
            )
            axs[ch, sample_num].set_xticks([])
            axs[ch, sample_num].set_yticks([])
            axs[ch, sample_num].set_title(f"sample {sample_num}")
            plt.colorbar(cp, ax=axs[ch, sample_num])

    return fig


def create_map(mask, steady_mask, output, gt, num_samples=3):
    if steady_mask is not None:
        mask += steady_mask
        mask[mask < 0] = 0
        mask[mask > 1] = 1
        assert (
            (mask == 0) | (mask == 1)
        ).all(), "Not all values in mask are zeros or ones!"

    samples = torch.randint(output.shape[0], (num_samples,))

    fig, axs = plt.subplots(
        2, num_samples, squeeze=False, figsize=(num_samples * 7, 14)
    )

    gt_ch = gt[:, :, 0, :, :]
    output_ch = output[:, :, 0, :, :]

    for sample_num in range(num_samples):
        target = gt_ch[samples[sample_num]].squeeze()
        pred = output_ch[samples[sample_num]].squeeze()

        vmin, vmax = torch.quantile(
            target, torch.tensor([0.05, 0.95], device=target.device)
        )

        cp1 = axs[0, sample_num].matshow(
            target.cpu(), cmap="RdBu_r", vmin=vmin, vmax=vmax
        )
        cp2 = axs[1, sample_num].matshow(
            pred.cpu(), cmap="RdBu_r", vmin=vmin, vmax=vmax
        )

        plt.colorbar(cp1, ax=axs[0, sample_num])
        plt.colorbar(cp2, ax=axs[1, sample_num])

        axs[0, sample_num].set_xticks([])
        axs[0, sample_num].set_yticks([])
        axs[1, sample_num].set_xticks([])
        axs[1, sample_num].set_yticks([])
        axs[0, sample_num].set_title(f"gt - sample {sample_num}")
        axs[1, sample_num].set_title(f"output - sample {sample_num}")
    return fig


def get_all_error_distributions(
    mask, steady_mask, output, gt, domain="valid", num_samples=1000
):
    error_dists = [
        calculate_error_distributions(
            mask,
            steady_mask,
            output,
            gt,
            operation=op,
            domain=domain,
            num_samples=num_samples,
        )
        for op in ["E", "AE", "RE", "RAE"]
    ]
    return error_dists


def get_all_error_maps(mask, steady_mask, output, gt, num_samples=3):
    error_maps = [
        create_error_map(
            mask,
            steady_mask,
            output,
            gt,
            num_samples=num_samples,
            operation=op,
            domain="valid",
        )
        for op in ["E", "AE", "RE", "RAE"]
    ]
    return error_maps


def create_ensemble_timeseries(
    gt,
    output,
    reference=None,
    lats=None,
    lons=None,
    time=None,
    lat_range=None,
    lon_range=None,
    title="Ensemble Timeseries",
    save_path=None,
):
    """
    Create timeseries plot with ensemble mean, std, and optional reference data.

    Args:
        gt: np.ndarray or torch.Tensor, shape (ens, time, lat, lon)
        output: np.ndarray or torch.Tensor, shape (ens, time, lat, lon)
        reference: np.ndarray or torch.Tensor, shape (time, lat, lon), optional
        lats: 1D array of latitude values, shape (lat,), optional
        lons: 1D array of longitude values, shape (lon,), optional
        time: 1D array of time values (e.g., years), shape (time,), optional
        lat_range: tuple (lat_min, lat_max) to cut region, optional
        lon_range: tuple (lon_min, lon_max) to cut region, optional
        title: Title for the plot
        save_path: If provided, saves the figure to this path
    """
    # Convert to numpy if torch tensor
    if hasattr(gt, "detach"):
        gt = gt.detach().cpu().numpy()
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    if reference is not None and hasattr(reference, "detach"):
        reference = reference.detach().cpu().numpy()
    if lats is not None and hasattr(lats, "detach"):
        lats = lats.detach().cpu().numpy()
    if lons is not None and hasattr(lons, "detach"):
        lons = lons.detach().cpu().numpy()
    if time is not None:
        if hasattr(time, "detach"):
            time = time.detach().cpu().numpy()
        elif hasattr(time, "values"):
            # Handle xarray DataArray
            time = time.values

    # Cut region if lat/lon ranges are provided
    if lat_range is not None or lon_range is not None:
        if lats is None or lons is None:
            raise ValueError(
                "lats and lons must be provided when using lat_range or lon_range"
            )

        # Find indices for lat range
        if lat_range is not None:
            lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
            lat_indices = np.where(lat_mask)[0]
        else:
            lat_indices = np.arange(len(lats))

        # Find indices for lon range
        if lon_range is not None:
            lon_mask = (lons >= lon_range[0]) & (lons <= lon_range[1])
            lon_indices = np.where(lon_mask)[0]
        else:
            lon_indices = np.arange(len(lons))

        # Cut the data
        gt = gt[
            :,
            :,
            lat_indices[0] : lat_indices[-1] + 1,
            lon_indices[0] : lon_indices[-1] + 1,
        ]
        output = output[
            :,
            :,
            lat_indices[0] : lat_indices[-1] + 1,
            lon_indices[0] : lon_indices[-1] + 1,
        ]
        if reference is not None:
            reference = reference[
                :,
                lat_indices[0] : lat_indices[-1] + 1,
                lon_indices[0] : lon_indices[-1] + 1,
            ]

    # Compute spatial mean for each ensemble member and timestep
    gt_spatial_mean = np.nanmean(gt, axis=(2, 3))  # shape: (ens, time)
    output_spatial_mean = np.nanmean(output, axis=(2, 3))  # shape: (ens, time)

    # Compute ensemble mean and std across ensemble members
    gt_mean = np.nanmean(gt_spatial_mean, axis=0)  # shape: (time,)
    gt_std = np.nanstd(gt_spatial_mean, axis=0)  # shape: (time,)
    output_mean = np.nanmean(output_spatial_mean, axis=0)  # shape: (time,)
    output_std = np.nanstd(output_spatial_mean, axis=0)  # shape: (time,)

    # Use provided time array or default to time steps
    if time is not None:
        time_steps = time
        xlabel = "Time"
    else:
        time_steps = np.arange(len(gt_mean))
        xlabel = "Time Step"

    # Calculate RMSE and correlation if reference is provided
    rmse_val = None
    corr_val = None
    if reference is not None:
        ref_spatial_mean = np.nanmean(reference, axis=(1, 2))  # shape: (time,)

        # Calculate RMSE between output ensemble mean and reference
        rmse_val = np.sqrt(np.nanmean((output_mean - ref_spatial_mean) ** 2))
        rmse_gt = np.sqrt(np.nanmean((gt_mean - ref_spatial_mean) ** 2))

        # Calculate correlation between output ensemble mean and reference
        valid_mask = ~(np.isnan(output_mean) | np.isnan(ref_spatial_mean))
        if valid_mask.sum() > 0:
            corr_val = np.corrcoef(
                output_mean[valid_mask], ref_spatial_mean[valid_mask]
            )[0, 1]
            corr_gt = np.corrcoef(gt_mean[valid_mask], ref_spatial_mean[valid_mask])[
                0, 1
            ]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot GT ensemble mean and std
    ax.plot(time_steps, gt_mean, color="blue", linewidth=2, label="GT Mean")
    ax.fill_between(
        time_steps,
        gt_mean - gt_std,
        gt_mean + gt_std,
        color="blue",
        alpha=0.3,
        label="GT ±1 Std",
    )

    # Plot Output ensemble mean and std
    ax.plot(time_steps, output_mean, color="red", linewidth=2, label="Output Mean")
    ax.fill_between(
        time_steps,
        output_mean - output_std,
        output_mean + output_std,
        color="red",
        alpha=0.3,
        label="Output ±1 Std",
    )

    # Plot reference data if provided
    if reference is not None:
        ax.plot(
            time_steps,
            ref_spatial_mean,
            color="green",
            linewidth=2,
            linestyle="--",
            label="Reference",
        )

    # Update title with RMSE and correlation if available
    if rmse_val is not None and corr_val is not None:
        fig_title = f"{title} (RMSE: ML: {rmse_val:.2e}, GT: {rmse_gt:.2e}, Corr: ML: {corr_val:.2f}, GT: {corr_gt:.2f})"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Spatial Mean Value")
    ax.set_title(fig_title)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(
            f"{save_path}/images/{title.replace(' ', '_')}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)
    else:
        plt.show()


def create_example_maps(
    gt,
    output,
    mask=None,
    reference=None,
    land_mask=None,
    lats=None,
    lons=None,
    num_timesteps=10,
    title="Example Timeseries Maps",
    save_path=None,
):
    """
    Create maps showing first N timesteps of GT, output, and ensemble means, plus time means in last row.

    Args:
        gt: np.ndarray or torch.Tensor, shape (ens, time, lat, lon)
        output: np.ndarray or torch.Tensor, shape (ens, time, lat, lon)
        mask: np.ndarray or torch.Tensor, shape (ens, time, lat, lon), optional binary mask
        reference: np.ndarray or torch.Tensor, shape (time, lat, lon), optional
        land_mask: np.ndarray or torch.Tensor, shape (lat, lon), optional - masks land areas with NaN
        num_timesteps: Number of timesteps to plot (default: 10)
        title: Title for the plot
        save_path: If provided, saves the figure to this path
    """
    # Convert to numpy if torch tensor
    if hasattr(gt, "detach"):
        gt = gt.detach().cpu().numpy()
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    if mask is not None and hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    if reference is not None and hasattr(reference, "detach"):
        reference = reference.detach().cpu().numpy()
    if land_mask is not None and hasattr(land_mask, "detach"):
        land_mask = land_mask.detach().cpu().numpy()

    n_ens, n_time, nlat, nlon = gt.shape
    num_timesteps = min(num_timesteps, n_time)

    # Apply land mask if provided
    if land_mask is not None:
        # Expand land_mask to match dimensions and apply
        gt = np.where(land_mask[np.newaxis, np.newaxis, :, :], np.nan, gt)
        output = np.where(land_mask[np.newaxis, np.newaxis, :, :], np.nan, output)
        if reference is not None:
            reference = np.where(land_mask[np.newaxis, :, :], np.nan, reference)

    # Calculate ensemble means
    gt_ens_mean = np.nanmean(gt, axis=0)  # shape: (time, lat, lon)
    output_ens_mean = np.nanmean(output, axis=0)  # shape: (time, lat, lon)

    # Calculate mask minimum over ensemble dimension if mask provided
    if mask is not None:
        mask_min = np.min(mask, axis=0)  # shape: (time, lat, lon)

    # Determine number of rows: num_timesteps + 1 (for time mean)
    n_rows = num_timesteps + 1
    n_cols = 5 if reference is not None else 4

    # Create figure with cartopy projection
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(title, fontsize=20)

    # Compute global vmin/vmax for consistent color scale
    all_data = [
        gt[0, :num_timesteps],
        output[0, :num_timesteps],
        gt_ens_mean[:num_timesteps],
        output_ens_mean[:num_timesteps],
    ]
    if reference is not None:
        all_data.append(reference[:num_timesteps])
    vmin = np.nanmin([np.nanquantile(d, 0.02) for d in all_data[0:1]])
    vmax = np.nanmax([np.nanquantile(d, 0.98) for d in all_data[0:1]])

    # Plot first num_timesteps
    for t in range(num_timesteps):
        # GT - First ensemble member
        ax0 = fig.add_subplot(
            n_rows, n_cols, t * n_cols + 1, projection=ccrs.PlateCarree()
        )
        im0 = ax0.imshow(
            gt[0, t],
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            extent=[min(lons), max(lons), min(lats), max(lats)],
            transform=ccrs.PlateCarree(),
        )
        ax0.coastlines(linewidth=0.5, color="black")
        ax0.set_extent(
            [min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree()
        )
        if mask is not None:
            y_coords, x_coords = np.where(mask[0, t] == 1)
            ax0.scatter(
                x_coords * 360 / nlon + min(lons),
                y_coords * 180 / nlat + min(lats),
                s=0.1,
                c="black",
                alpha=0.5,
                marker=".",
                transform=ccrs.PlateCarree(),
            )
        ax0.set_title(f"GT Member 1 - T{t+1}")
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # Output - First ensemble member
        ax1 = fig.add_subplot(
            n_rows, n_cols, t * n_cols + 2, projection=ccrs.PlateCarree()
        )
        im1 = ax1.imshow(
            output[0, t],
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            extent=[min(lons), max(lons), min(lats), max(lats)],
            transform=ccrs.PlateCarree(),
        )
        ax1.coastlines(linewidth=0.5, color="black")
        ax1.set_extent(
            [min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree()
        )
        if mask is not None:
            y_coords, x_coords = np.where(mask[0, t] == 1)
            ax1.scatter(
                x_coords * 360 / nlon + min(lons),
                y_coords * 180 / nlat + min(lats),
                s=0.1,
                c="black",
                alpha=0.5,
                marker=".",
                transform=ccrs.PlateCarree(),
            )
        ax1.set_title(f"Output Member 1 - T{t+1}")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # GT Ensemble Mean
        ax2 = fig.add_subplot(
            n_rows, n_cols, t * n_cols + 3, projection=ccrs.PlateCarree()
        )
        im2 = ax2.imshow(
            gt_ens_mean[t],
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            extent=[min(lons), max(lons), min(lats), max(lats)],
            transform=ccrs.PlateCarree(),
        )
        ax2.coastlines(linewidth=0.5, color="black")
        ax2.set_extent(
            [min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree()
        )
        if mask is not None:
            y_coords, x_coords = np.where(mask_min[t] == 1)
            ax2.scatter(
                x_coords * 360 / nlon + min(lons),
                y_coords * 180 / nlat + min(lats),
                s=0.1,
                c="black",
                alpha=0.5,
                marker=".",
                transform=ccrs.PlateCarree(),
            )
        ax2.set_title(f"GT Ens Mean - T{t+1}")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Output Ensemble Mean
        ax3 = fig.add_subplot(
            n_rows, n_cols, t * n_cols + 4, projection=ccrs.PlateCarree()
        )
        im3 = ax3.imshow(
            output_ens_mean[t],
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            extent=[min(lons), max(lons), min(lats), max(lats)],
            transform=ccrs.PlateCarree(),
        )
        ax3.coastlines(linewidth=0.5, color="black")
        ax3.set_extent(
            [min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree()
        )
        if mask is not None:
            y_coords, x_coords = np.where(mask_min[t] == 1)
            ax3.scatter(
                x_coords * 360 / nlon + min(lons),
                y_coords * 180 / nlat + min(lats),
                s=0.1,
                c="black",
                alpha=0.5,
                marker=".",
                transform=ccrs.PlateCarree(),
            )
        ax3.set_title(f"Output Ens Mean - T{t+1}")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Reference (if provided)
        if reference is not None:
            ax4 = fig.add_subplot(
                n_rows, n_cols, t * n_cols + 5, projection=ccrs.PlateCarree()
            )
            im4 = ax4.imshow(
                reference[t],
                origin="lower",
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                extent=[min(lons), max(lons), min(lats), max(lats)],
                transform=ccrs.PlateCarree(),
            )
            ax4.coastlines(linewidth=0.5, color="black")
            ax4.set_extent(
                [min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree()
            )
            if mask is not None:
                y_coords, x_coords = np.where(mask_min[t] == 1)
                ax4.scatter(
                    x_coords * 360 / nlon + min(lons),
                    y_coords * 180 / nlat + min(lats),
                    s=0.1,
                    c="black",
                    alpha=0.5,
                    marker=".",
                    transform=ccrs.PlateCarree(),
                )
            ax4.set_title(f"Reference - T{t+1}")
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Last row: Time means
    gt_time_mean_member1 = np.nanmean(gt[0], axis=0)
    output_time_mean_member1 = np.nanmean(output[0], axis=0)
    gt_ens_time_mean = np.nanmean(gt_ens_mean, axis=0)
    output_ens_time_mean = np.nanmean(output_ens_mean, axis=0)

    if mask is not None:
        mask_time_mean_member1 = np.min(mask[0], axis=0)
        mask_time_mean_ens = np.min(mask_min, axis=0)

    row_idx = num_timesteps

    ax_mean0 = fig.add_subplot(
        n_rows, n_cols, row_idx * n_cols + 1, projection=ccrs.PlateCarree()
    )
    im_mean0 = ax_mean0.imshow(
        gt_time_mean_member1,
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        extent=[min(lons), max(lons), min(lats), max(lats)],
        transform=ccrs.PlateCarree(),
    )
    ax_mean0.coastlines(linewidth=0.5, color="black")
    ax_mean0.set_extent(
        [min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree()
    )
    if mask is not None:
        y_coords, x_coords = np.where(mask_time_mean_member1 == 1)
        ax_mean0.scatter(
            x_coords * 360 / nlon + min(lons),
            y_coords * 180 / nlat + min(lats),
            s=0.1,
            c="black",
            alpha=0.5,
            marker=".",
            transform=ccrs.PlateCarree(),
        )
    ax_mean0.set_title("GT Member 1 - Time Mean")
    plt.colorbar(im_mean0, ax=ax_mean0, fraction=0.046, pad=0.04)

    ax_mean1 = fig.add_subplot(
        n_rows, n_cols, row_idx * n_cols + 2, projection=ccrs.PlateCarree()
    )
    im_mean1 = ax_mean1.imshow(
        output_time_mean_member1,
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        extent=[min(lons), max(lons), min(lats), max(lats)],
        transform=ccrs.PlateCarree(),
    )
    ax_mean1.coastlines(linewidth=0.5, color="black")
    ax_mean1.set_extent(
        [min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree()
    )
    if mask is not None:
        y_coords, x_coords = np.where(mask_time_mean_member1 == 1)
        ax_mean1.scatter(
            x_coords * 360 / nlon + min(lons),
            y_coords * 180 / nlat + min(lats),
            s=0.1,
            c="black",
            alpha=0.5,
            marker=".",
            transform=ccrs.PlateCarree(),
        )
    ax_mean1.set_title("Output Member 1 - Time Mean")
    plt.colorbar(im_mean1, ax=ax_mean1, fraction=0.046, pad=0.04)

    ax_mean2 = fig.add_subplot(
        n_rows, n_cols, row_idx * n_cols + 3, projection=ccrs.PlateCarree()
    )
    im_mean2 = ax_mean2.imshow(
        gt_ens_time_mean,
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        extent=[min(lons), max(lons), min(lats), max(lats)],
        transform=ccrs.PlateCarree(),
    )
    ax_mean2.coastlines(linewidth=0.5, color="black")
    ax_mean2.set_extent(
        [min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree()
    )
    if mask is not None:
        y_coords, x_coords = np.where(mask_time_mean_ens == 1)
        ax_mean2.scatter(
            x_coords * 360 / nlon + min(lons),
            y_coords * 180 / nlat + min(lats),
            s=0.1,
            c="black",
            alpha=0.5,
            marker=".",
            transform=ccrs.PlateCarree(),
        )
    ax_mean2.set_title("GT Ens Mean - Time Mean")
    plt.colorbar(im_mean2, ax=ax_mean2, fraction=0.046, pad=0.04)

    ax_mean3 = fig.add_subplot(
        n_rows, n_cols, row_idx * n_cols + 4, projection=ccrs.PlateCarree()
    )
    im_mean3 = ax_mean3.imshow(
        output_ens_time_mean,
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        extent=[min(lons), max(lons), min(lats), max(lats)],
        transform=ccrs.PlateCarree(),
    )
    ax_mean3.coastlines(linewidth=0.5, color="black")
    ax_mean3.set_extent(
        [min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree()
    )
    if mask is not None:
        y_coords, x_coords = np.where(mask_time_mean_ens == 1)
        ax_mean3.scatter(
            x_coords * 360 / nlon + min(lons),
            y_coords * 180 / nlat + min(lats),
            s=0.1,
            c="black",
            alpha=0.5,
            marker=".",
            transform=ccrs.PlateCarree(),
        )
    ax_mean3.set_title("Output Ens Mean - Time Mean")
    plt.colorbar(im_mean3, ax=ax_mean3, fraction=0.046, pad=0.04)

    if reference is not None:
        ref_time_mean = np.nanmean(reference, axis=0)
        ax_mean4 = fig.add_subplot(
            n_rows, n_cols, row_idx * n_cols + 5, projection=ccrs.PlateCarree()
        )
        im_mean4 = ax_mean4.imshow(
            ref_time_mean,
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            extent=[min(lons), max(lons), min(lats), max(lats)],
            transform=ccrs.PlateCarree(),
        )
        ax_mean4.coastlines(linewidth=0.5, color="black")
        ax_mean4.set_extent(
            [min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree()
        )
        if mask is not None:
            y_coords, x_coords = np.where(mask_time_mean_ens == 1)
            ax_mean4.scatter(
                x_coords * 360 / nlon + min(lons),
                y_coords * 180 / nlat + min(lats),
                s=0.1,
                c="black",
                alpha=0.5,
                marker=".",
                transform=ccrs.PlateCarree(),
            )
        ax_mean4.set_title("Reference - Time Mean")
        plt.colorbar(im_mean4, ax=ax_mean4, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        plt.savefig(
            f"{save_path}/images/{title.replace(' ', '_')}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)
    else:
        plt.show()

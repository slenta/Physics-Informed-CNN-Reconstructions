import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import embed


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
    gt, output, reference=None, title="Ensemble Timeseries", save_path=None
):
    """
    Create timeseries plot with ensemble mean, std, and optional reference data.

    Args:
        gt: np.ndarray or torch.Tensor, shape (ens, time, lat, lon)
        output: np.ndarray or torch.Tensor, shape (ens, time, lat, lon)
        reference: np.ndarray or torch.Tensor, shape (time, lat, lon), optional
        title: Title for the plot
        save_path: If provided, saves the figure to this path
    """

    # Compute spatial mean for each ensemble member and timestep
    gt_spatial_mean = np.nanmean(gt, axis=(2, 3))  # shape: (ens, time)
    output_spatial_mean = np.nanmean(output, axis=(2, 3))  # shape: (ens, time)

    # Compute ensemble mean and std across ensemble members
    gt_mean = np.nanmean(gt_spatial_mean, axis=0)  # shape: (time,)
    gt_std = np.nanstd(gt_spatial_mean, axis=0)  # shape: (time,)
    output_mean = np.nanmean(output_spatial_mean, axis=0)  # shape: (time,)
    output_std = np.nanstd(output_spatial_mean, axis=0)  # shape: (time,)

    time_steps = np.arange(len(gt_mean))

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
        label="GT Std",
    )

    # Plot Output ensemble mean and std
    ax.plot(time_steps, output_mean, color="red", linewidth=2, label="Output Mean")
    ax.fill_between(
        time_steps,
        output_mean - output_std,
        output_mean + output_std,
        color="red",
        alpha=0.3,
        label="Output Std",
    )

    # Plot reference data if provided
    if reference is not None:
        ref_spatial_mean = np.nanmean(reference, axis=(1, 2))  # shape: (time,)
        ax.plot(
            time_steps,
            ref_spatial_mean,
            color="green",
            linewidth=2,
            linestyle="--",
            label="Reference",
        )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Spatial Mean Value")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(
            f"{save_path}/images/{title.replace(' ', '_')}.png", bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()


def create_example_maps(
    gt,
    output,
    mask=None,
    reference=None,
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

    n_ens, n_time, nlat, nlon = gt.shape
    num_timesteps = min(num_timesteps, n_time)

    # Calculate ensemble means
    gt_ens_mean = np.nanmean(gt, axis=0)  # shape: (time, lat, lon)
    output_ens_mean = np.nanmean(output, axis=0)  # shape: (time, lat, lon)

    # Calculate mask minimum over ensemble dimension if mask provided
    if mask is not None:
        mask_min = np.min(mask, axis=0)  # shape: (time, lat, lon)

    # Determine number of rows: num_timesteps + 1 (for time mean)
    n_rows = num_timesteps + 1
    n_cols = 5 if reference is not None else 4

    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )
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
    vmin = np.nanmin([np.nanquantile(d, 0.02) for d in all_data])
    vmax = np.nanmax([np.nanquantile(d, 0.98) for d in all_data])

    # Plot first num_timesteps
    for t in range(num_timesteps):
        # GT - First ensemble member
        im0 = axes[t, 0].imshow(
            gt[1, t], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
        )
        if mask is not None:
            y_coords, x_coords = np.where(mask[0, t] == 1)
            axes[t, 0].scatter(
                x_coords,
                y_coords,
                s=0.01,
                c="black",
                alpha=0.8,
                marker="o",
                linewidths=0.5,
            )
        axes[t, 0].set_title(f"GT Member 1 - T{t+1}")
        axes[t, 0].set_ylabel(f"Time {t+1}")
        axes[t, 0].set_xticks([])
        axes[t, 0].set_yticks([])
        fig.colorbar(im0, ax=axes[t, 0], fraction=0.046, pad=0.04)

        # Output - First ensemble member
        im1 = axes[t, 1].imshow(
            output[0, t], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
        )
        if mask is not None:
            y_coords, x_coords = np.where(mask[0, t] == 1)
            axes[t, 1].scatter(
                x_coords,
                y_coords,
                s=0.01,
                c="black",
                alpha=0.8,
                marker="o",
                linewidths=0.5,
            )
        axes[t, 1].set_title(f"Output Member 1 - T{t+1}")
        axes[t, 1].set_xticks([])
        axes[t, 1].set_yticks([])
        fig.colorbar(im1, ax=axes[t, 1], fraction=0.046, pad=0.04)

        # GT Ensemble Mean
        im2 = axes[t, 2].imshow(
            gt_ens_mean[t], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
        )
        if mask is not None:
            y_coords, x_coords = np.where(mask_min[t] == 1)
            axes[t, 2].scatter(
                x_coords,
                y_coords,
                s=0.01,
                c="black",
                alpha=0.8,
                marker="o",
                linewidths=0.5,
            )
        axes[t, 2].set_title(f"GT Ens Mean - T{t+1}")
        axes[t, 2].set_xticks([])
        axes[t, 2].set_yticks([])
        fig.colorbar(im2, ax=axes[t, 2], fraction=0.046, pad=0.04)

        # Output Ensemble Mean
        im3 = axes[t, 3].imshow(
            output_ens_mean[t], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
        )
        if mask is not None:
            y_coords, x_coords = np.where(mask_min[t] == 1)
            axes[t, 3].scatter(
                x_coords,
                y_coords,
                s=0.01,
                c="black",
                alpha=0.8,
                marker="o",
                linewidths=0.5,
            )
        axes[t, 3].set_title(f"Output Ens Mean - T{t+1}")
        axes[t, 3].set_xticks([])
        axes[t, 3].set_yticks([])
        fig.colorbar(im3, ax=axes[t, 3], fraction=0.046, pad=0.04)

        # Reference (if provided)
        if reference is not None:
            im4 = axes[t, 4].imshow(
                reference[t], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
            )
            if mask is not None:
                y_coords, x_coords = np.where(mask_min[t] == 1)
                axes[t, 4].scatter(
                    x_coords,
                    y_coords,
                    s=0.01,
                    c="black",
                    alpha=0.8,
                    marker="o",
                    linewidths=0.5,
                )
            axes[t, 4].set_title(f"Reference - T{t+1}")
            axes[t, 4].set_xticks([])
            axes[t, 4].set_yticks([])
            fig.colorbar(im4, ax=axes[t, 4], fraction=0.046, pad=0.04)

    # Last row: Time means
    gt_time_mean_member1 = np.nanmean(gt[0], axis=0)  # shape: (lat, lon)
    output_time_mean_member1 = np.nanmean(output[0], axis=0)  # shape: (lat, lon)
    gt_ens_time_mean = np.nanmean(gt_ens_mean, axis=0)  # shape: (lat, lon)
    output_ens_time_mean = np.nanmean(output_ens_mean, axis=0)  # shape: (lat, lon)

    # Mask time means
    if mask is not None:
        mask_time_mean_member1 = np.min(mask[0], axis=0)  # shape: (lat, lon)
        mask_time_mean_ens = np.min(mask_min, axis=0)  # shape: (lat, lon)

    im_mean0 = axes[-1, 0].imshow(
        gt_time_mean_member1, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
    )
    if mask is not None:
        y_coords, x_coords = np.where(mask_time_mean_member1 == 1)
        axes[-1, 1].scatter(
            x_coords,
            y_coords,
            s=0.01,
            c="black",
            alpha=0.8,
            marker="o",
            linewidths=0.5,
        )
    axes[-1, 0].set_title("GT Member 1 - Time Mean")
    axes[-1, 0].set_ylabel("Time Mean")
    axes[-1, 0].set_xticks([])
    axes[-1, 0].set_yticks([])
    fig.colorbar(im_mean0, ax=axes[-1, 0], fraction=0.046, pad=0.04)

    im_mean1 = axes[-1, 1].imshow(
        output_time_mean_member1, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
    )
    if mask is not None:
        y_coords, x_coords = np.where(mask_time_mean_member1 == 1)
        axes[-1, 1].scatter(
            x_coords,
            y_coords,
            s=0.01,
            c="black",
            alpha=0.8,
            marker="o",
            linewidths=0.5,
        )
    axes[-1, 1].set_title("Output Member 1 - Time Mean")
    axes[-1, 1].set_xticks([])
    axes[-1, 1].set_yticks([])
    fig.colorbar(im_mean1, ax=axes[-1, 1], fraction=0.046, pad=0.04)

    im_mean2 = axes[-1, 2].imshow(
        gt_ens_time_mean, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
    )
    if mask is not None:
        y_coords, x_coords = np.where(mask_time_mean_ens == 1)
        axes[-1, 2].scatter(
            x_coords,
            y_coords,
            s=0.01,
            c="black",
            alpha=0.8,
            marker="o",
            linewidths=0.5,
        )
    axes[-1, 2].set_title("GT Ens Mean - Time Mean")
    axes[-1, 2].set_xticks([])
    axes[-1, 2].set_yticks([])
    fig.colorbar(im_mean2, ax=axes[-1, 2], fraction=0.046, pad=0.04)

    im_mean3 = axes[-1, 3].imshow(
        output_ens_time_mean, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
    )
    if mask is not None:
        y_coords, x_coords = np.where(mask_time_mean_ens == 1)
        axes[-1, 3].scatter(
            x_coords,
            y_coords,
            s=0.01,
            c="black",
            alpha=0.8,
            marker="o",
            linewidths=0.5,
        )
    axes[-1, 3].set_title("Output Ens Mean - Time Mean")
    axes[-1, 3].set_xticks([])
    axes[-1, 3].set_yticks([])
    fig.colorbar(im_mean3, ax=axes[-1, 3], fraction=0.046, pad=0.04)

    if reference is not None:
        ref_time_mean = np.nanmean(reference, axis=0)  # shape: (lat, lon)
        im_mean4 = axes[-1, 4].imshow(
            ref_time_mean, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
        )
        if mask is not None:
            y_coords, x_coords = np.where(mask_time_mean_ens == 1)
            axes[-1, 4].scatter(
                x_coords,
                y_coords,
                s=0.01,
                c="black",
                alpha=0.8,
                marker="o",
                linewidths=0.5,
            )
        axes[-1, 4].set_title("Reference - Time Mean")
        axes[-1, 4].set_xticks([])
        axes[-1, 4].set_yticks([])
        fig.colorbar(im_mean4, ax=axes[-1, 4], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        plt.savefig(
            f"{save_path}/images/{title.replace(' ', '_')}.png", bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()

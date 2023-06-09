from cProfile import label
import matplotlib
import os
import h5py
import cartopy.crs as ccrs
from isort import file
from matplotlib.pyplot import title
import numpy as np
import pylab as plt
import config as cfg
import evaluation_og as evalu
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, norm

if not os.path.exists(f"../Asi_maskiert/pdfs/validation/{cfg.save_part}/"):
    os.makedirs(f"../Asi_maskiert/pdfs/validation/{cfg.save_part}/")


def uncertainty_plot(part, iteration, length=764, del_t=1):
    if cfg.val_cut:
        val_cut = "_cut"
    else:
        val_cut = ""

    hc_a_it, hc_o_it = evalu.ml_ensemble_iteration(
        16, part, iteration=iteration - 4000, length=length
    )
    hc_a, hc_o = evalu.hc_ml_ensemble(16, part, iteration=iteration, length=length)

    del_a = np.std(hc_a, axis=0)
    del_it = np.std(hc_a_it, axis=0)

    hc_a_it, hc_a, del_a, del_it = (
        evalu.running_mean_std(hc_a_it, "mean", del_t),
        evalu.running_mean_std(hc_a, "mean", del_t),
        evalu.running_mean_std(del_a, "mean", del_t),
        evalu.running_mean_std(del_it, "mean", del_t),
    )

    f = h5py.File(
        f"{cfg.val_dir}{part}/timeseries_{str(iteration)}_assimilation_anhang_{cfg.eval_im_year}{val_cut}.hdf5",
        "r",
    )

    assi = f.get("net_ts")
    hc_gt = np.array(f.get("gt_ts"))

    assi, hc_gt = (
        evalu.running_mean_std(assi, "mean", del_t),
        evalu.running_mean_std(hc_gt, "mean", del_t),
    )

    start = 1958 + (del_t // 12) // 2
    end = 2022 - (del_t // 12) // 2

    length = len(hc_gt)
    ticks = np.arange(0, length, 12 * 5)
    labels = np.arange(start, end, 5)

    plt.figure(figsize=(10, 6))
    # plt.plot(en4, label="EN4 Reanalysis", color="purple")
    plt.fill_between(
        range(len(assi)),
        assi + del_a + del_it + del_it / 2,
        assi - del_a - del_it - del_it / 2,
        label="Ensemble Spread NN Training Instance",
        color="lightsteelblue",
    )
    plt.fill_between(
        range(len(assi)),
        assi + del_a + del_it,
        assi - del_a - del_it,
        label="Ensemble Spread Stopping Point",
        color="cornflowerblue",
    )
    plt.fill_between(
        range(len(assi)),
        assi + del_a,
        assi - del_a,
        label="Ensemble Spread Reconstruction",
        color="royalblue",
    )
    plt.plot(hc_gt, label="Assimilation Heat Content", color="darkred")
    plt.plot(assi, label="Network Reconstructed Heat Content", color="navy")

    plt.grid()
    plt.legend()
    plt.xticks(ticks=ticks, labels=labels)
    plt.title("SPG OHC Estimates and Uncertainties")
    plt.xlabel("Time in years")
    plt.ylabel("Heat Content [J]")
    plt.savefig(
        f"../Asi_maskiert/pdfs/validation/{part}/uncertainty_{str(iteration)}_{cfg.eval_im_year}{val_cut}.pdf"
    )
    plt.show()


def masked_output_vis(part, iter, time, depth):
    fa = h5py.File(
        f"../Asi_maskiert/results/validation/{part}/validation_{iter}_assimilation_{cfg.mask_argo}_{cfg.eval_im_year}.hdf5",
        "r",
    )
    fo = h5py.File(
        f"../Asi_maskiert/results/validation/{part}/validation_{iter}_observations_{cfg.mask_argo}_{cfg.eval_im_year}.hdf5",
        "r",
    )
    fm = h5py.File("../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5", "r")

    continent_mask = np.array(fm.get("continent_mask"))[:, :]

    mask = np.array(fa.get("mask")[time, depth, :, :]) * continent_mask
    output_a = np.array(fa.get("output")[time, depth, :, :]) * continent_mask
    image_a = np.array(fa.get("image")[time, depth, :, :]) * continent_mask
    output_o = np.array(fo.get("output")[time, depth, :, :]) * continent_mask
    image_o = np.array(fo.get("image")[time, depth, :, :]) * continent_mask

    mask_grey = np.where(mask == 0, np.NaN, mask) * continent_mask
    save_dir = f"../Asi_maskiert/pdfs/validation/{part}/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    fig.suptitle("Anomaly North Atlantic SSTs")
    plt.subplot(2, 2, 1)
    plt.title(f"Masked Assimilations")
    current_cmap = plt.cm.coolwarm
    current_cmap.set_bad(color="gray")
    plt.imshow(
        image_a * mask_grey,
        cmap=current_cmap,
        vmin=-3,
        vmax=3,
        aspect="auto",
        interpolation=None,
    )
    plt.xlabel("Transformed Longitudes")
    plt.ylabel("Transformed Latitudes")
    # plt.colorbar(label='Temperature in °C')
    plt.subplot(2, 2, 2)
    plt.title(f"Observation Mask")
    im2 = plt.imshow(
        image_o * mask_grey,
        cmap="coolwarm",
        vmin=-3,
        vmax=3,
        aspect="auto",
        interpolation=None,
    )
    plt.xlabel("Transformed Longitudes")
    plt.ylabel("Transformed Latitudes")
    # plt.colorbar(label='Temperature in °C')
    plt.subplot(2, 2, 3)
    plt.title("Masked Assimilation Reconstructions")
    plt.imshow(output_a * mask_grey, vmin=-3, vmax=3, cmap="coolwarm", aspect="auto")
    plt.xlabel("Transformed Longitudes")
    plt.ylabel("Transformed Latitudes")
    # plt.colorbar(label='Temperature in °C')
    plt.subplot(2, 2, 4)
    plt.title("Masked Observations Reconstructions")
    plt.imshow(output_o * mask_grey, cmap="coolwarm", vmin=-3, vmax=3, aspect="auto")
    plt.xlabel("Transformed Longitudes")
    plt.ylabel("Transformed Latitudes")
    plt.colorbar(label="Annomaly Correlation")
    fig.savefig(
        f"../Asi_maskiert/pdfs/validation/{part}/validation_masked_{iter}_timestep_{str(time)}_depth_{str(depth)}.pdf",
        dpi=fig.dpi,
    )
    plt.show()


def output_vis(part, iter, time, depth, mode):
    if mode == "Assimilation":
        f = h5py.File(
            f"../Asi_maskiert/results/validation/{part}/validation_{iter}_assimilation_{cfg.mask_argo}_{cfg.eval_im_year}.hdf5",
            "r",
        )
    elif mode == "Observations":
        f = h5py.File(
            f"../Asi_maskiert/results/validation/{part}/validation_{iter}_observations_{cfg.mask_argo}_{cfg.eval_im_year}.hdf5",
            "r",
        )

    fm = h5py.File("../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5", "r")
    continent_mask = np.array(fm.get("continent_mask"))

    gt = np.array(f.get("gt")[:, depth, :, :]) * continent_mask
    mask = np.array(f.get("mask")[:, depth, :, :]) * continent_mask
    output = np.array(f.get("output")[:, depth, :, :]) * continent_mask
    image = np.array(f.get("image")[:, depth, :, :]) * continent_mask

    mask_grey = np.where(mask == 0, np.NaN, mask) * continent_mask
    T_mean_output = np.nanmean(np.nanmean(output, axis=1), axis=1)
    T_mean_gt = np.nanmean(np.nanmean(gt, axis=1), axis=1)

    if depth <= 5:
        limit = 2
    elif 5 < depth <= 15:
        limit = 1.5
    elif depth > 15:
        limit = 1

    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    fig.suptitle("Anomaly North Atlantic SSTs")
    plt.subplot(1, 3, 1)
    plt.title(f"Masked Image {mode}")
    current_cmap = plt.cm.coolwarm
    current_cmap.set_bad(color="gray")
    im1 = plt.imshow(
        image[time, :, :] * mask_grey[time, :, :],
        cmap=current_cmap,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation=None,
    )
    plt.xlabel("Transformed Longitudes")
    plt.ylabel("Transformed Latitudes")
    # plt.colorbar(label='Temperature in °C')
    plt.subplot(1, 3, 2)
    plt.title(f"Reconstructed {mode}: Network Output")
    im2 = plt.imshow(
        output[time, :, :],
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation=None,
    )
    plt.xlabel("Transformed Longitudes")
    plt.ylabel("Transformed Latitudes")
    plt.colorbar(label="Temperature in °C")
    plt.subplot(1, 3, 3)
    plt.title("Original Assimilation Image")
    plt.imshow(gt[time, :, :], vmin=-limit, vmax=limit, cmap="coolwarm", aspect="auto")
    plt.xlabel("Transformed Longitudes")
    plt.ylabel("Transformed Latitudes")
    plt.colorbar(label="Temperature in °C")
    fig.savefig(
        f"../Asi_maskiert/pdfs/validation/{part}/validation_{mode}_{iter}_timestep_{str(time)}_depth_{str(depth)}_{cfg.eval_im_year}.pdf",
        dpi=fig.dpi,
    )
    plt.show()


def correlation_plotting(
    path, iteration, obs=False, starts=[0, 552], ends=[551, 764], mask_argo="anhang"
):
    f_a = h5py.File(
        f"../Asi_maskiert/results/validation/{path}/validation_{iteration}_assimilation_{mask_argo}_{cfg.eval_im_year}.hdf5",
        "r",
    )
    f_o = h5py.File(
        f"../Asi_maskiert/results/validation/{path}/validation_{iteration}_observations_{mask_argo}_{cfg.eval_im_year}.hdf5",
        "r",
    )
    f_ah = h5py.File(
        f"../Asi_maskiert/results/validation/{path}/heatcontent_{iteration}_assimilation_{mask_argo}_{cfg.eval_im_year}.hdf5",
        "r",
    )
    f_oh = h5py.File(
        f"../Asi_maskiert/results/validation/{path}/heatcontent_{iteration}_observations_{mask_argo}_{cfg.eval_im_year}.hdf5",
        "r",
    )

    fm = h5py.File("../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5", "r")
    continent_mask = np.array(fm.get("coastlines"))
    f_spg = h5py.File(f"{cfg.mask_dir}SPG_Maske.hdf5")
    spg = f_spg.get("SPG")

    hc_a = np.nan_to_num(np.array(f_ah.get("hc_net")), nan=0)
    hc_gt = np.nan_to_num(np.array(f_oh.get("hc_gt")), nan=0)
    hc_o = np.nan_to_num(np.array(f_oh.get("hc_net")), nan=0)

    # cut time of correlation
    correlation_a_1 = evalu.correlation(
        hc_a[starts[0] : ends[0]], hc_gt[starts[0] : ends[0]]
    )
    correlation_a_2 = evalu.correlation(
        hc_a[starts[1] : ends[1]], hc_gt[starts[1] : ends[1]]
    )
    correlation_o_1 = evalu.correlation(
        hc_o[starts[0] : ends[0]], hc_gt[starts[0] : ends[0]]
    )
    correlation_o_2 = evalu.correlation(
        hc_o[starts[1] : ends[1]], hc_gt[starts[1] : ends[1]]
    )
    if obs == True:
        corr_1 = correlation_o_1
        corr_2 = correlation_o_2
    else:
        corr_1 = correlation_a_1
        corr_2 = correlation_a_2

    # correlation_argo_a, sig_argo_a = evalu.correlation(hc_a[552:], hc_gt[552:])
    # correlation_preargo_a, sig_preargo_a = evalu.correlation(hc_a[:552], hc_gt[:552])
    # correlation_argo_o, sig_argo_o = evalu.correlation(hc_o[552:], hc_gt[552:])
    # correlation_preargo_o, sig_preargo_o = evalu.correlation(hc_o[:552], hc_gt[:552])

    acc_a_1 = pearsonr(
        np.nanmean(hc_gt[starts[0] : ends[0]], axis=(2, 1)),
        np.nanmean(hc_a[starts[0] : ends[0]], axis=(2, 1)),
    )[0]
    acc_a_2 = pearsonr(
        np.nanmean(np.nanmean(hc_gt[starts[1] : ends[1]], axis=1), axis=1),
        np.nanmean(np.nanmean(hc_a[starts[1] : ends[1]], axis=1), axis=1),
    )[0]
    acc_o_1 = pearsonr(
        np.nanmean(np.nanmean(hc_gt[starts[0] : ends[0]], axis=1), axis=1),
        np.nanmean(np.nanmean(hc_o[starts[0] : ends[0]], axis=1), axis=1),
    )[0]
    acc_o_2 = pearsonr(
        np.nanmean(np.nanmean(hc_gt[starts[1] : ends[1]], axis=1), axis=1),
        np.nanmean(np.nanmean(hc_o[starts[1] : ends[1]], axis=1), axis=1),
    )[0]

    if obs == True:
        corr_1 = correlation_o_1
        corr_2 = correlation_o_2
        acc_1 = acc_o_1
        acc_2 = acc_o_2
    else:
        corr_1 = correlation_a_1
        corr_2 = correlation_a_2
        acc_1 = acc_a_1
        acc_2 = acc_a_2

    corr_1 = np.nan_to_num(corr_1, nan=0)
    corr_2 = np.nan_to_num(corr_2, nan=0)

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Correlation: Assimilation - Neural Network Reconstruction")
    plt.subplot(1, 2, 1)
    plt.title(f"Correlation: 1958 -- 1968")
    current_cmap = plt.cm.coolwarm
    current_cmap.set_bad(color="gray")
    plt.imshow(
        corr_1 * spg * continent_mask, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto"
    )
    plt.xlabel("Transformed Longitudes")
    plt.ylabel("Transformed Latitudes")
    plt.subplot(1, 2, 2)
    plt.title(f"Correlation: 1968 -- 2004")
    current_cmap = plt.cm.coolwarm
    current_cmap.set_bad(color="gray")
    plt.imshow(
        corr_2 * spg * continent_mask, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto"
    )
    plt.xlabel("Transformed Longitudes")
    plt.ylabel("Transformed Latitudes")
    plt.colorbar(label="Annomaly Correlation")
    fig.savefig(
        f"../Asi_maskiert/pdfs/validation/{path}/correlation_{iteration}_{cfg.eval_im_year}_{starts[0]}_{ends[1]}.pdf",
        dpi=fig.dpi,
    )
    plt.show()

    # fig = plt.figure(figsize=(13, 12), constrained_layout=True)
    # fig.suptitle("Anomaly North Atlantic Heat Content")
    # plt.subplot(2, 2, 1)
    # plt.title(f"Argo: Assimilation Reconstruction Correlation: {acc_mean_argo_a:.2f}")
    # current_cmap = plt.cm.coolwarm
    # current_cmap.set_bad(color="gray")
    # plt.scatter(sig_argo_a[1], sig_argo_a[0], c="black", s=0.7, marker=".", alpha=0.2)
    # im3 = plt.imshow(
    #    correlation_argo_a, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto"
    # )
    # plt.xlabel("Transformed Longitudes")
    # plt.ylabel("Transformed Latitudes")
    # plt.subplot(2, 2, 2)
    # plt.title(f"Argo: Observation Reconstruction Correlation: {acc_mean_argo_o:.2f}")
    # current_cmap = plt.cm.coolwarm
    # current_cmap.set_bad(color="gray")
    # plt.scatter(sig_argo_o[1], sig_argo_o[0], c="black", s=0.7, marker=".", alpha=0.2)
    # im3 = plt.imshow(
    #    correlation_argo_o, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto"
    # )
    # plt.xlabel("Transformed Longitudes")
    # plt.ylabel("Transformed Latitudes")
    # plt.subplot(2, 2, 3)
    # plt.title(
    #    f"Preargo: Assimilation Reconstruction Correlation: {acc_mean_preargo_a:.2f}"
    # )
    # current_cmap = plt.cm.coolwarm
    # current_cmap.set_bad(color="gray")
    # plt.scatter(
    #    sig_preargo_a[1], sig_preargo_a[0], c="black", s=0.7, marker=".", alpha=0.2
    # )
    # im3 = plt.imshow(
    #    correlation_preargo_a, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto"
    # )
    # plt.xlabel("Transformed Longitudes")
    # plt.ylabel("Transformed Latitudes")
    # plt.subplot(2, 2, 4)
    # plt.title(
    #    f"Preargo: Observations Reconstruction Correlation: {acc_mean_preargo_o:.2f}"
    # )
    # current_cmap = plt.cm.coolwarm
    # current_cmap.set_bad(color="gray")
    # plt.scatter(
    #    sig_preargo_o[1], sig_preargo_o[0], c="black", s=0.7, marker=".", alpha=0.2
    # )
    # im3 = plt.imshow(
    #    correlation_preargo_o, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto"
    # )
    # plt.xlabel("Transformed Longitudes")
    # plt.ylabel("Transformed Latitudes")
    # plt.colorbar(label="Annomaly Correlation")
    # fig.savefig(
    #    f"../Asi_maskiert/pdfs/validation/{path}/correlation_{iteration}_{cfg.eval_im_year}.pdf",
    #    dpi=fig.dpi,
    # )
    # plt.show()


def hc_plotting(path, iteration, time=600, obs=False, mask_argo="anhang"):
    f = h5py.File(
        f"{cfg.val_dir}{path}/heatcontent_{str(iteration)}_assimilation_{mask_argo}_{cfg.eval_im_year}{cfg.attribute_anomaly}.hdf5",
        "r",
    )
    fo = h5py.File(
        f"{cfg.val_dir}{path}/heatcontent_{str(iteration)}_observations_{mask_argo}_{cfg.eval_im_year}{cfg.attribute_anomaly}.hdf5",
        "r",
    )
    fc = h5py.File(
        f"{cfg.val_dir}{path}/validation_{str(iteration)}_assimilation_{mask_argo}_{cfg.eval_im_year}.hdf5",
        "r",
    )
    fc_o = h5py.File(
        f"{cfg.val_dir}{path}/validation_{str(iteration)}_observations_{mask_argo}_{cfg.eval_im_year}.hdf5",
        "r",
    )
    f_en4 = h5py.File(f"{cfg.im_dir}En4_1950_2021_NA_newgrid.hdf5")

    f_cm = h5py.File(f"{cfg.mask_dir}Kontinent_newgrid.hdf5")
    f_spg = h5py.File(f"{cfg.mask_dir}SPG_Maske.hdf5")
    spg = f_spg.get("SPG")
    continent_mask = np.array(f_cm.get("continent_mask"))
    coastlines = np.array(f_cm.get("coastlines"))

    image_o = np.array(fc_o.get("image"))
    image = np.array(fc.get("image"))
    mask = np.nan_to_num(np.array(fc.get("mask")), nan=0)

    if type(time) == int:
        image = image[time, :, :, :]
        # image = evalu.heat_content_single(image)
        # image_grey = np.where(image == 0, np.nan, image) * continent_mask

        # image_o = image_o[time, :, :, :]
        # image_o = evalu.heat_content_single(image_o)
        # image_grey_o = np.where(image_o == 0, np.nan, image_o) * continent_mask

        hc_assi = np.nan_to_num(np.array(f.get("hc_net"))[time, :, :], nan=1)
        hc_gt = np.nan_to_num(np.array(f.get("hc_gt"))[time, :, :], nan=1)
        hc_obs = np.nan_to_num(np.array(fo.get("hc_net"))[time, :, :], nan=1)
        en4 = np.nan_to_num(np.array(f_en4.get("ohc"))[time, :, :], nan=1)
        en4_mask = en4 * mask[time, 0, :, :]
    else:
        time_1 = time[0]
        time_2 = time[1]
        image = np.nanmean(image[time_1:time_2, :, :, :], axis=0)
        # image = evalu.heat_content_single(image)
        # image_grey = image * continent_mask

        # image_o = np.nanmean(image_o[time_1:time_2, :, :, :], axis=0)
        # image_o = evalu.heat_content_single(image_o)
        # image_grey_o = image_o * continent_mask

        hc_assi = np.nanmean(
            np.nan_to_num(np.array(f.get("hc_net"))[time_1:time_2, :, :], nan=1),
            axis=0,
        )
        hc_gt = np.nanmean(
            np.nan_to_num(np.array(f.get("hc_gt"))[time_1:time_2, :, :], nan=1),
            axis=0,
        )
        hc_obs = np.nanmean(
            np.nan_to_num(np.array(fo.get("hc_net"))[time_1:time_2, :, :], nan=1),
            axis=0,
        )
        en4 = np.nanmean(
            np.nan_to_num(np.array(f_en4.get("ohc"))[time_1:time_2, :, :], nan=1),
            axis=0,
        )
        en4_mask = en4 * mask[(time_1 + time_2) // 2, 0, :, :]
    cmap_1 = plt.cm.get_cmap("coolwarm").copy()
    cmap_1.set_bad(color="darkgrey")
    cmap_2 = plt.cm.get_cmap("bwr").copy()
    cmap_2.set_bad(color="black")
    if cfg.attribute_anomaly == "_full":
        mini = 1.5e10
        maxi = 2.5e10
    else:
        mini = -3e9
        maxi = 3e9

    lines = np.nan_to_num(spg * coastlines, nan=3)
    line = np.where(lines == 3)
    mask_nan = np.where(mask == 0, np.nan, 1)

    if obs == False:
        fig = plt.figure(figsize=(16, 7), constrained_layout=True)
        fig.suptitle("North Atlantic Heat Content Comparison")
        plt.subplot(1, 3, 1)
        plt.title(f"Assimilation Mask")
        im1 = plt.imshow(
            mask_nan[60, 0, :, :] * hc_gt * coastlines * spg * continent_mask,
            cmap=cmap_1,
            vmin=mini,
            vmax=maxi,
            aspect="auto",
            interpolation=None,
        )
        plt.scatter(line[1], line[0], c="black", s=15, alpha=1)
        plt.xlabel("Transformed Longitudes")
        plt.ylabel("Transformed Latitudes")
        plt.subplot(1, 3, 2)
        plt.title("Assimilation Heat Content")
        plt.imshow(
            coastlines * spg * hc_gt * continent_mask,
            cmap=cmap_1,
            vmin=mini,
            vmax=maxi,
            aspect="auto",
            interpolation=None,
        )
        plt.scatter(line[1], line[0], c="black", s=15, alpha=1)
        plt.xlabel("Transformed Longitudes")
        plt.ylabel("Transformed Latitudes")
        plt.subplot(1, 3, 3)
        plt.title("Network Output Heat Content")
        plt.imshow(
            hc_assi * spg * coastlines * continent_mask,
            cmap=cmap_1,
            vmin=mini,
            vmax=maxi,
            aspect="auto",
            interpolation=None,
        )
        plt.scatter(line[1], line[0], c="black", s=15, alpha=1)
        plt.xlabel("Transformed Longitudes")
        plt.ylabel("Transformed Latitudes")
        plt.colorbar(mappable=im1, label="Heat Content in J")
        plt.savefig(
            f"../Asi_maskiert/pdfs/validation/{path}/heat_content_{time}_{iteration}_{mask_argo}_{cfg.eval_im_year}{cfg.attribute_anomaly}.pdf",
            dpi=fig.dpi,
        )
        plt.show()

    else:
        fig = plt.figure(figsize=(16, 7), constrained_layout=True)
        fig.suptitle("North Atlantic Heat Content Comparison")
        plt.subplot(1, 3, 1)
        plt.title(f"EN4 Reanalysis at Points of Observation")
        plt.imshow(
            en4_mask * coastlines * spg,
            cmap=cmap_1,
            vmin=mini,
            vmax=maxi,
            aspect="auto",
            interpolation=None,
        )
        plt.xlabel("Transformed Longitudes")
        plt.ylabel("Transformed Latitudes")
        plt.subplot(1, 3, 2)
        plt.title("Assimilation Heat Content")
        plt.imshow(
            hc_gt * spg * coastlines,
            cmap=cmap_1,
            vmin=mini,
            vmax=maxi,
            aspect="auto",
            interpolation=None,
        )
        plt.xlabel("Transformed Longitudes")
        plt.ylabel("Transformed Latitudes")
        plt.subplot(1, 3, 3)
        plt.title("Network Output Heat Content")
        plt.imshow(
            hc_obs * spg * coastlines,
            cmap=cmap_1,
            vmin=mini,
            vmax=maxi,
            aspect="auto",
            interpolation=None,
        )
        plt.xlabel("Transformed Longitudes")
        plt.ylabel("Transformed Latitudes")
        plt.colorbar(label="Heat Content in J")
        plt.savefig(
            f"../Asi_maskiert/pdfs/validation/{path}/heat_content_{time}_{iteration}_{mask_argo}_{cfg.eval_im_year}_obs{cfg.attribute_anomaly}.pdf",
            dpi=fig.dpi,
        )
        plt.show()


def plot_val_error(part, iteration, interval, combine_start, in_channels):
    error_overall = np.zeros(shape=(in_channels, iteration // interval - 1))
    print(iteration // interval - 1, iteration, interval)
    for save_part in np.arange(combine_start, combine_start + in_channels):
        file = f"{cfg.val_dir}part_{save_part}/val_errors.hdf5"
        f = h5py.File(file, "r")
        rsmes = np.array(f.get("rsmes")).flatten()
        error_overall[save_part - combine_start, :] = rsmes

    print(error_overall.shape)
    error = np.nanmean(error_overall, axis=0)
    print(error)
    xx = np.arange(interval, iteration, interval)

    plt.figure(figsize=(8, 6))
    plt.title("Validation Error")
    plt.xlabel("Training Iteration")
    plt.ylabel("Validation Error")
    plt.plot(xx, error)
    plt.grid()
    plt.savefig(f"../Asi_maskiert/pdfs/validation/{part}/valerror_{iteration}.pdf")
    plt.show()

    return np.where(min(error))


def timeseries_plotting(
    part,
    iteration,
    obs=True,
    del_t=1,
    argo="full",
    mask_argo="full",
    compare=False,
    single=False,
    summary=False,
):
    if cfg.val_cut:
        val_cut = "_cut"
    else:
        val_cut = ""
    if compare == True:
        compar = "_compare"
    else:
        compar = ""
    if single == True:
        singl = "_single"
    else:
        singl = ""
    if summary == True:
        summ = "_summary"
    else:
        summ = ""

    f_a = h5py.File(
        f"{cfg.val_dir}/{part}/validation_{iteration}_assimilation_{mask_argo}_{cfg.eval_im_year}{val_cut}.hdf5",
        "r",
    )
    f_o = h5py.File(
        f"{cfg.val_dir}{part}/validation_{iteration}_observations_{mask_argo}_{cfg.eval_im_year}{val_cut}.hdf5",
        "r",
    )
    f = h5py.File(
        f"{cfg.val_dir}{part}/timeseries_{iteration}_assimilation_{mask_argo}_{cfg.eval_im_year}{val_cut}.hdf5",
        "r",
    )
    fo = h5py.File(
        f"{cfg.val_dir}{part}/timeseries_{iteration}_observations_{mask_argo}_{cfg.eval_im_year}{val_cut}.hdf5",
        "r",
    )
    # f_a = h5py.File(
    #     f"{cfg.val_dir}/{part}/validation_{iteration}_assimilation_full.hdf5",
    #     "r",
    # )
    # f_o = h5py.File(
    #     f"{cfg.val_dir}{part}/validation_{iteration}_observations_full.hdf5",
    #     "r",
    # )
    # f = h5py.File(
    #     f"{cfg.val_dir}{part}/timeseries_{str(iteration)}_assimilation_full.hdf5",
    #     "r",
    # )
    # fo = h5py.File(
    #     f"{cfg.val_dir}{part}/timeseries_{str(iteration)}_observations_full.hdf5",
    #     "r",
    # )
    f_en4 = h5py.File(f"{cfg.im_dir}En4_reanalysis_1950_2020_NA{val_cut}.hdf5")
    f_iap = h5py.File(
        f"{cfg.im_dir}/IAP/IAP_2000m_1958_2021_NA_newgrid_newdepth{val_cut}.hdf5"
    )

    iap = np.array(f_iap.get("ohc"))
    iap = np.nansum(iap, axis=(1, 2))
    en4 = np.array(f_en4.get("ohc"))
    en4 = np.nansum(en4, axis=(1, 2))

    hc_assi = np.array(f.get("net_ts"))
    hc_gt = np.array(f.get("gt_ts"))
    hc_assi_masked = np.array(f.get("net_ts_masked"))
    hc_gt_masked = np.array(f.get("gt_ts_masked"))
    hc_obs = np.array(fo.get("net_ts"))
    hc_obs_masked = np.array(fo.get("net_ts_masked"))
    hc_obs_gt_masked = np.array(fo.get("gt_ts_masked"))

    print(hc_assi.shape, hc_obs.shape)

    gt = np.array(f_a.get("gt")[:, 0, :, :])
    continent_mask = np.where(gt == 0, np.NaN, 1)
    gt = gt * continent_mask

    # calculate uncertainty through ensemble standard deviations
    std_a = evalu.running_mean_std(hc_assi, mode="std", del_t=del_t)
    std_o = evalu.running_mean_std(hc_obs, mode="std", del_t=del_t)
    std_gt = evalu.running_mean_std(hc_gt, mode="std", del_t=del_t)

    gt_mean, std_gt, hc_all = evalu.hc_ensemble_mean_std(
        cfg.im_dir, name="Image_r", members=16
    )
    hc_all_a, hc_all_o = evalu.hc_ml_ensemble(
        members=15, part="part_18", iteration=585000, length=764
    )
    del_a = evalu.running_mean_std(
        np.nanstd(hc_all_a, axis=0), mode="mean", del_t=del_t
    )
    del_o = evalu.running_mean_std(
        np.nanstd(hc_all_a, axis=0), mode="mean", del_t=del_t
    )
    std_gt = evalu.running_mean_std(std_gt, mode="mean", del_t=del_t)
    del_gt = std_gt[: len(std_a)]

    # calculate running mean, if necessary
    if del_t != 1:
        hc_assi = evalu.running_mean_std(hc_assi, mode="mean", del_t=del_t)
        hc_gt = evalu.running_mean_std(hc_gt, mode="mean", del_t=del_t)
        hc_obs = evalu.running_mean_std(hc_obs, mode="mean", del_t=del_t)
        en4 = evalu.running_mean_std(en4, mode="mean", del_t=del_t)
        iap = evalu.running_mean_std(iap, mode="mean", del_t=del_t)
        hc_assi_masked = evalu.running_mean_std(
            hc_assi_masked, mode="mean", del_t=del_t
        )
        hc_gt_masked = evalu.running_mean_std(hc_gt_masked, mode="mean", del_t=del_t)
        hc_obs_masked = evalu.running_mean_std(hc_obs_masked, mode="mean", del_t=del_t)
        hc_obs_gt_masked = evalu.running_mean_std(
            hc_obs_gt_masked, mode="mean", del_t=del_t
        )

    # define running mean timesteps
    if argo == "argo":
        hc_gt, hc_obs, hc_assi, del_a, del_o, del_gt, en4, iap = (
            hc_gt[552:754],
            hc_obs[552:754],
            hc_assi[552:754],
            del_a[552:754],
            del_o[552:754],
            del_gt[552:754],
            en4[552:754],
            iap[552:754],
        )
        # start = 2004 + (del_t // 12) // 2
        # end = 2021 - (del_t // 12) // 2
        start = 1958
        end = 2021
    elif argo == "full":
        hc_gt, hc_obs, hc_assi, del_a, del_o, del_gt, en4, iap = (
            hc_gt[:754],
            hc_obs[:754],
            hc_assi[:754],
            del_a[:754],
            del_o[:754],
            del_gt[:754],
            en4[:754],
            iap[:754],
        )
        start = 1958 + (del_t // 12) // 2
        end = 2021 - (del_t // 12) // 2
    elif argo == "anhang":
        hc_gt, hc_obs, hc_assi, del_a, del_o, del_gt, en4, iap = (
            hc_gt[552:],
            hc_obs[552:],
            hc_assi[552:],
            del_a[552:],
            del_o[552:],
            del_gt[552:],
            en4[552:],
            iap[552:],
        )
        start = 2004 + (del_t // 12) // 2
        end = 2022 - (del_t // 12) // 2

    else:
        start = 1958 + (del_t // 12) // 2
        end = 2021 - (del_t // 12) // 2

    length = len(hc_gt)
    if argo == "argo":
        ticks = np.arange(-46 * 12, length, 12 * 5)
        labels = np.arange(start, end, 5)  #
        print(ticks.shape, labels.shape)
    else:
        ticks = np.arange(0, length, 12 * 5)
        labels = np.arange(start, end, 5)  #

    plt.figure(figsize=(10, 6))
    if argo == "anhang":
        plt.axvline(x=201, color="red")
    if obs == True:
        print(pearsonr(hc_gt, hc_obs)[0])
        o = "_obs"
        if compare == True:
            plt.plot(en4, label="EN4 Objective Analysis Heat Content", color="purple")
            # plt.plot(iap, label="IAP Assimilation Heat Content", color="orange")
            print(
                pearsonr(iap, hc_obs[: len(iap)])[0],
                pearsonr(en4, hc_obs[: len(en4)])[0],
            )

        plt.plot(hc_gt, label="Assimilation OHC", color="darkred")
        plt.fill_between(
            range(len(hc_gt)),
            hc_gt + del_gt,
            hc_gt - del_gt,
            label="Ensemble Spread Assimilation",
            color="lightcoral",
        )
        if single == False:
            plt.plot(hc_obs, label="Neural Network OHC", color="royalblue")
            plt.fill_between(
                range(len(hc_obs)),
                hc_obs + del_o,
                hc_obs - del_o,
                label="Ensemble Spread Neural Network",
                color="lightsteelblue",
            )
    else:
        print(pearsonr(hc_gt, hc_assi)[0])
        o = ""
        plt.plot(hc_gt, label="Assimilation Heat Content", color="darkred")
        plt.fill_between(
            range(len(hc_assi)),
            hc_gt + del_gt,
            hc_gt - del_gt,
            label="Ensemble Spread Assimilation",
            color="lightcoral",
        )
        if compare == True:
            plt.plot(en4, label="EN4 Objective Analysis Heat Content", color="purple")
        if single == False:
            plt.plot(
                hc_assi, label="Network Reconstructed Heat Content", color="royalblue"
            )
            plt.fill_between(
                range(len(hc_assi)),
                hc_assi + del_a,
                hc_assi - del_a,
                label="Ensemble Spread Reconstruction",
                color="lightsteelblue",
            )
        if summary == True:
            plt.plot(hc_obs, label="Directly Reconstructed Observations", color="grey")
            plt.fill_between(
                range(len(hc_obs)),
                hc_obs + del_o,
                hc_obs - del_o,
                label="Ensemble Spread Reconstruction",
                color="lightgrey",
            )

    plt.grid()
    plt.legend(loc=9)
    plt.xticks(ticks=ticks, labels=labels)
    plt.title("SPG OHC Estimates")
    plt.xlabel("Time in years")
    plt.ylabel("Heat Content [J]")
    plt.savefig(
        f"../Asi_maskiert/pdfs/validation/{part}/validation_timeseries_{str(iteration)}_{del_t}_mean{val_cut}{o}_{cfg.eval_im_year}_mask_{mask_argo}_data_{argo}{singl}{compar}{summ}.pdf"
    )
    plt.show()

    # plt.figure(figsize=(10, 6))

    # plt.plot(hc_assi_masked, label="NN reconstruction", color="blue")
    # plt.plot(hc_gt_masked, label="Assimilation Heat Content", color="darkred")
    # plt.plot(hc_obs_masked, label="Direct NN reconstruction", color="grey")
    # plt.plot(hc_obs_gt_masked, label="Observations Heat Content", color="purple")
    # plt.grid()
    # plt.legend()
    # plt.xticks(ticks=ticks, labels=labels)
    # plt.title(
    #     "Comparison Reconstruction to Assimilation Timeseries at Observation Points"
    # )
    # plt.xlabel("Time in years")
    # plt.ylabel("Heat Content [J]")
    # plt.savefig(f"../Asi_maskiert/pdfs/validation/{part}/masked_1_obs.pdf")
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # if argo == "anhang":
    #    plt.axvline(x=201, color="red")
    # if obs == False:
    #    misfit = hc_assi - hc_gt
    #    del_total = del_a + del_gt
    #    plt.plot(misfit, label="Misfit: Assimilation and Indirect NN Reconstruction")
    #    plt.fill_between(
    #        range(len(misfit)),
    #        misfit + del_total,
    #        misfit - del_total,
    #        label="Combined Uncertainty",
    #        color="lightsteelblue",
    #    )
    # else:
    #    misfit = hc_obs - hc_gt
    #    del_total = del_o + del_gt
    #    plt.plot(misfit, label="Misfit: Assimilation and Direct NN Reconstruction")
    #    plt.fill_between(
    #        range(len(misfit)),
    #        misfit + del_total,
    #        misfit - del_total,
    #        label="Combined Uncertainty",
    #        color="lightsteelblue",
    #    )
    # plt.grid()
    # plt.legend()
    # plt.xticks(ticks=ticks, labels=labels)
    # plt.title("Misfit: Neural Network Reconstruction - Assimilation")
    # plt.xlabel("Time in years")
    # plt.ylabel("Heat Content [J]")
    # plt.savefig(
    #    f"../Asi_maskiert/pdfs/validation/{part}/misfit_{str(iteration)}_{del_t}_mean{val_cut}{o}_{cfg.eval_im_year}_mask_{mask_argo}_data_{argo}.pdf"
    # )
    # plt.show()


def std_plotting(del_t, ensemble=True):
    if cfg.val_cut:
        val_cut = "_cut"
        f = h5py.File(
            f"{cfg.val_dir}{cfg.save_part}/timeseries_{cfg.resume_iter}_assimilation_{cfg.mask_argo}_{cfg.eval_im_year}_cut.hdf5",
            "r",
        )
        fo = h5py.File(
            f"{cfg.val_dir}{cfg.save_part}/timeseries_{cfg.resume_iter}_observations_{cfg.mask_argo}_{cfg.eval_im_year}_cut.hdf5",
            "r",
        )
    else:
        val_cut = ""
        f = h5py.File(
            f"{cfg.val_dir}{cfg.save_part}/timeseries_{cfg.resume_iter}_assimilation_{cfg.mask_argo}_{cfg.eval_im_year}.hdf5",
            "r",
        )
        fo = h5py.File(
            f"{cfg.val_dir}{cfg.save_part}/timeseries_{cfg.resume_iter}_observations_{cfg.mask_argo}_{cfg.eval_im_year}.hdf5",
            "r",
        )

    hc_network = np.array(f.get("net_ts"))
    hc_gt = np.array(f.get("gt_ts"))
    hc_obs = np.array(fo.get("net_ts"))

    # plot running std
    std_a = evalu.running_mean_std(hc_network, mode="std", del_t=del_t)
    std_gt = evalu.running_mean_std(hc_gt, mode="std", del_t=del_t)
    std_o = evalu.running_mean_std(hc_obs, mode="std", del_t=del_t)

    if ensemble == True:
        gt_mean, std_gt, hc_all = evalu.hc_ensemble_mean_std(
            cfg.im_dir, name="Image_r", members=16
        )
        std_gt = evalu.running_mean_std(std_gt, mode="mean", del_t=del_t)
        std_gt = std_gt[: len(std_a)]

    length = len(std_gt)

    # define running mean timesteps
    if del_t != 1:
        start = 1958 + (del_t // 12) // 2
        end = 2021 - (del_t // 12) // 2
        ticks = np.arange(0, length, 12 * 5)
        labels = np.arange(start, end, 5)
    else:
        ticks = np.arange(0, length, 12 * 5)
        labels = np.arange(1958, 2021, 5)

    plt.figure(figsize=(10, 6))
    plt.plot(std_a, label="Standard Deviation Reconstruction")
    plt.plot(std_gt, label="Standard Deviation Assimilation")
    # plt.plot(hc_obs_std, label='Standard Deviation Observations Reconstruction')
    plt.grid()
    plt.legend()
    plt.xticks(ticks=ticks, labels=labels)
    plt.title(
        "Comparison Standard Devation of Reconstructions to Original Assimilation"
    )
    plt.xlabel("Time in years")
    plt.ylabel(f"Standard Deviation of Heat Content ({str(del_t/12)} years)")
    plt.savefig(
        f"../Asi_maskiert/pdfs/validation/{cfg.save_part}/validation_std_timeseries_{cfg.mask_argo}_{cfg.eval_im_year}_{cfg.resume_iter}_{str(del_t)}{val_cut}.pdf"
    )
    plt.show()


def pattern_corr_plot(
    part, del_t=1, obs=False, resume_iter=550000, argo="full", mask_argo="full"
):
    if cfg.val_cut:
        val_cut = "_cut"
    else:
        val_cut = ""

    f_o = h5py.File(
        f"{cfg.val_dir}{part}/pattern_corr_ts_{resume_iter}_observations_{mask_argo}_{cfg.eval_im_year}_mean_{del_t}{val_cut}.hdf5",
        "r",
    )
    f_a = h5py.File(
        f"{cfg.val_dir}{part}/pattern_corr_ts_{resume_iter}_assimilation_{mask_argo}_{cfg.eval_im_year}_mean_{del_t}{val_cut}.hdf5",
        "r",
    )

    corr_o = np.squeeze(np.array(f_o.get("corr_ts")))
    corr_a = np.squeeze(np.array(f_a.get("corr_ts")))

    if argo == "argo":
        corr_o, corr_a = corr_o[552:754], corr_a[552:754]
        start = 2004 + (del_t // 12) // 2
        end = 2021 - (del_t // 12) // 2
    elif argo == "anhang":
        corr_o, corr_a = corr_o[552:], corr_a[552:]
        start = 2004 + (del_t // 12) // 2
        end = 2022 - (del_t // 12) // 2
    else:
        start = 1958 + (del_t // 12) // 2
        end = 2021 - (del_t // 12) // 2

    # calculate running mean, if necessary
    if del_t != 1:
        length = len(corr_o)

        ticks = np.arange(0, length, 12 * 5)
        labels = np.arange(start, end, 5)

    else:
        length = len(corr_o)
        ticks = np.arange(0, length, 12 * 5)
        labels = np.arange(start, 2021, 5)

    print(np.nanmean(corr_o), np.nanmean(corr_a))

    plt.figure(figsize=(10, 6))
    if argo == "anhang":
        plt.axvline(x=201, color="red")
    if obs == True:
        o = "_obs"
        plt.plot(
            corr_o, label="Pattern Correlation Direct Reconstruction", color="grey"
        )
        plt.plot(
            corr_a, label="Pattern Correlation Indirect Reconstruction", color="red"
        )
    else:
        o = ""
        plt.plot(corr_a, label="Pattern Correlation Assimilation")
    plt.grid()
    plt.legend()
    plt.ylim(0, 1)
    plt.xticks(ticks=ticks, labels=labels)
    plt.title("OHC Pattern Correlation")
    plt.xlabel("Time in years")
    plt.ylabel(f"Pattern Correlation as ACC")
    plt.savefig(
        f"../Asi_maskiert/pdfs/validation/{part}/pattern_cor_timeseries_{resume_iter}_{del_t}_{cfg.eval_im_year}_{argo}{val_cut}{o}.pdf"
    )
    plt.show()


def error_pdf(argo, n_windows=1, del_t=1):
    f = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/timeseries_{cfg.resume_iter}_assimilation_{argo}.hdf5",
        "r",
    )
    fo = h5py.File(
        f"{cfg.val_dir}{cfg.save_part}/timeseries_{cfg.resume_iter}_observations_{argo}.hdf5",
        "r",
    )

    hc_assi = np.array(f.get("net_ts"))
    hc_gt = np.array(f.get("gt_ts"))
    hc_obs = np.array(fo.get("net_ts"))

    # calculate running mean, if necessary
    if del_t != 1:
        hc_assi = evalu.running_mean_std(hc_assi, mode="mean", del_t=del_t)
        hc_gt = evalu.running_mean_std(hc_gt, mode="mean", del_t=del_t)
        hc_obs = evalu.running_mean_std(hc_obs, mode="mean", del_t=del_t)

    if n_windows != 1:
        len_w = len(hc_assi) // n_windows
        for i in range(n_windows):
            globals()[f"hc_a_{str(i)}"] = hc_assi[len_w * i : len_w * (i + 1)]
            globals()[f"hc_o_{str(i)}"] = hc_obs[len_w * i : len_w * (i + 1)]
            globals()[f"hc_gt_{str(i)}"] = hc_gt[len_w * i : len_w * (i + 1)]

            globals()[f"error_a_{str(i)}"] = np.sqrt(
                (globals()[f"hc_a_{str(i)}"] - globals()[f"hc_gt_{str(i)}"]) ** 2
            )
            globals()[f"error_o_{str(i)}"] = np.sqrt(
                (globals()[f"hc_o_{str(i)}"] - globals()[f"hc_gt_{str(i)}"]) ** 2
            )

            globals()[f"pdf_a_{str(i)}"] = norm.pdf(
                np.sort(globals()[f"error_a_{str(i)}"]),
                np.mean(globals()[f"error_a_{str(i)}"]),
                np.std(globals()[f"error_a_{str(i)}"]),
            )
            globals()[f"pdf_o_{str(i)}"] = norm.pdf(
                np.sort(globals()[f"error_o_{str(i)}"]),
                np.mean(globals()[f"error_o_{str(i)}"]),
                np.std(globals()[f"error_o_{str(i)}"]),
            )

    plt.title("Error PDFs")
    if n_windows != 1:
        for i in range(n_windows):
            start = 1958 + del_t // 12 // 2 + i * (len_w // 12)
            end = start + len_w // 12
            if i == n_windows - 1:
                end = 2021
            plt.plot(
                np.sort(globals()[f"error_a_{str(i)}"]),
                globals()[f"pdf_a_{str(i)}"],
                label=f"Error Pdf: {start} -- {end}",
            )
            # plt.plot(np.sort(globals()[f'error_o_{str(i)}']), globals()[f'pdf_o_{str(i)}'], label=f'Observations Reconstruction Error Pdf {str(i)}')
    plt.grid()
    plt.ylabel("Probability Density")
    plt.xlabel("Absolute Error of Reconstruction")
    plt.legend()
    plt.savefig(f"../Asi_maskiert/pdfs/validation/{cfg.save_part}/error_pdfs.pdf")
    plt.show()


def new_4_plot(
    var_1,
    var_2,
    var_3,
    var_4,
    name_1,
    name_2,
    name_3,
    name_4,
    title,
    mini,
    maxi,
    cb_unit="Heat Content in J",
):
    # create datasets and cut versions for SPG highlighting
    var_1_cut = evalu.area_cutting_single(var_1)
    var_2_cut = evalu.area_cutting_single(var_2)
    var_3_cut = evalu.area_cutting_single(var_3)
    var_4_cut = evalu.area_cutting_single(var_4)

    var_1 = evalu.create_dataset(var_1, val_cut="")
    var_1_cut = evalu.create_dataset(var_1_cut, val_cut="_cut")
    var_2 = evalu.create_dataset(var_2, val_cut="")
    var_2_cut = evalu.create_dataset(var_2_cut, val_cut="_cut")
    var_3 = evalu.create_dataset(var_3, val_cut="")
    var_3_cut = evalu.create_dataset(var_3_cut, val_cut="_cut")
    var_4 = evalu.create_dataset(var_4, val_cut="")
    var_4_cut = evalu.create_dataset(var_4_cut, val_cut="_cut")

    # create colormap if necessary
    cmap_1 = plt.cm.get_cmap("coolwarm").copy()
    cmap_1.set_bad(color="black")

    # start figure layout and show variables
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    fig.suptitle(title, fontweight="bold", fontsize=15)
    ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax1.set_global()
    var_1.variable.plot.pcolormesh(
        ax=ax1,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=mini,
        vmax=maxi,
        alpha=0.4,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    var_1_cut.variable.plot.pcolormesh(
        ax=ax1,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=mini,
        vmax=maxi,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    ax1.coastlines()
    ax1.set_title(name_1)
    ax1.set_ylim([38, 72])
    ax1.set_xlim([-75, 5])
    gls1 = ax1.gridlines(color="lightgrey", linestyle="-", draw_labels=True)
    gls1.top_labels = False  # suppress top labels
    gls1.right_labels = False  # suppress right labels
    gls1.bottom_labels = False  # suppress bottom labels
    gls1.left_labels = False  # suppress left labels

    ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax2.set_global()
    var_2.variable.plot.pcolormesh(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=mini,
        vmax=maxi,
        alpha=0.4,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    im = var_2_cut.variable.plot.pcolormesh(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=mini,
        vmax=maxi,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    ax2.coastlines()
    ax2.set_title(name_2)
    ax2.set_ylim([38, 72])
    ax2.set_xlim([-75, 5])
    gls2 = ax2.gridlines(color="lightgrey", linestyle="-", draw_labels=True)
    gls2.top_labels = False  # suppress top labels
    gls2.right_labels = False  # suppress right labels
    gls2.bottom_labels = False  # suppress bottom labels
    gls2.left_labels = False  # suppress left labels
    cbar = plt.colorbar(im, shrink=0.8, location="bottom")
    cbar.set_label(cb_unit)

    ax3 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax3.set_global()
    var_3.variable.plot.pcolormesh(
        ax=ax3,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=mini,
        vmax=maxi,
        alpha=0.4,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    var_3_cut.variable.plot.pcolormesh(
        ax=ax3,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=mini,
        vmax=maxi,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    ax3.coastlines()
    ax3.set_title(name_3)
    ax3.set_ylim([38, 72])
    ax3.set_xlim([-75, 5])
    gls3 = ax3.gridlines(color="lightgrey", linestyle="-", draw_labels=True)
    gls3.top_labels = False  # suppress top labels
    gls3.right_labels = False  # suppress right labels
    gls3.bottom_labels = False  # suppress bottom labels
    gls3.left_labels = False  # suppress left labels

    ax4 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    ax4.set_global()
    var_4.variable.plot.pcolormesh(
        ax=ax4,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=mini,
        vmax=maxi,
        alpha=0.4,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    im = var_4_cut.variable.plot.pcolormesh(
        ax=ax4,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=mini,
        vmax=maxi,
        x="lon",
        y="lat",
        add_colorbar=False,
    )
    ax4.coastlines()
    ax4.set_title(name_4)
    ax4.set_ylim([38, 72])
    ax4.set_xlim([-75, 5])
    gls4 = ax4.gridlines(color="lightgrey", linestyle="-", draw_labels=True)
    gls4.top_labels = False  # suppress top labels
    gls4.right_labels = False  # suppress right labels
    gls4.bottom_labels = False  # suppress bottom labels
    gls4.left_labels = False  # suppress left labels
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label(cb_unit)
    plt.savefig(
        f"../Asi_maskiert/pdfs/validation/{cfg.save_part}/nw_images/{title}.pdf"
    )
    plt.show()

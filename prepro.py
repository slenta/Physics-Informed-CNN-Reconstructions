# script to preprocess necessary data for the nn input

from preprocessing import preprocessing
import config as cfg

cfg.set_train_args()


if cfg.prepro_mode == "image":
    dataset = preprocessing(
        cfg.im_dir,
        cfg.im_name,
        cfg.im_year,
        cfg.image_size,
        "image",
        cfg.in_channels,
    )
    dataset.save_data()
elif cfg.prepro_mode == "mask":
    dataset = preprocessing(
        cfg.mask_dir,
        cfg.mask_name,
        cfg.mask_year,
        cfg.image_size,
        "mask",
        cfg.in_channels,
    )
    dataset.save_data()
elif cfg.prepro_mode == "both":
    dataset = preprocessing(
        cfg.im_dir,
        cfg.im_name,
        cfg.im_year,
        cfg.image_size,
        "image",
        cfg.in_channels,
    )
    dataset1 = preprocessing(
        cfg.mask_dir,
        cfg.mask_name,
        cfg.mask_year,
        cfg.image_size,
        "mask",
        cfg.in_channels,
    )
    dataset.save_data()
    dataset1.save_data()
elif cfg.prepro_mode == "val":
    prepro_mask = preprocessing(
        cfg.mask_dir,
        cfg.mask_name,
        cfg.mask_year,
        cfg.image_size,
        "mask",
        cfg.in_channels,
    )
    prepro = preprocessing(
        cfg.im_dir,
        cfg.im_name,
        cfg.eval_im_year,
        cfg.image_size,
        "image",
        cfg.in_channels,
    )
    prepro_obs = preprocessing(
        cfg.mask_dir,
        cfg.mask_name,
        cfg.eval_mask_year,
        cfg.image_size,
        "val",
        cfg.in_channels,
    )
    prepro_obs.save_data()
    prepro_mask.save_data()
    prepro.save_data()
elif cfg.prepro_mode == "mixed":
    dataset = preprocessing(
        cfg.im_dir,
        cfg.im_name,
        cfg.im_year,
        cfg.image_size,
        "mixed",
        cfg.in_channels,
    )
    dataset.save_data()

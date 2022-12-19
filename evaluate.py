# Script to infill and evaluate certain model version


import matplotlib
import evaluation_og as evalu
from dataloader import MaskDataset
from dataloader import ValDataset
from preprocessing import preprocessing
from utils.io import load_ckpt, save_ckpt
import config as cfg
import torch
import numpy as np
from model.net import PConvLSTM
import os

matplotlib.use("Agg")
cfg.set_train_args()

if not os.path.exists(f"{cfg.save_dir}/images/{cfg.save_part}/"):
    os.makedirs(f"{cfg.save_dir}/images/{cfg.save_part}/")
if not os.path.exists(f"{cfg.save_dir}/ckpt/{cfg.save_part}/"):
    os.makedirs(f"{cfg.save_dir}/ckpt/{cfg.save_part}/")
if not os.path.exists(f"{cfg.save_dir}/validation/{cfg.save_part}/"):
    os.makedirs(f"{cfg.save_dir}/validation/{cfg.save_part}/")
print(f"{cfg.save_dir}/validation/{cfg.save_part}/")

if cfg.lstm_steps == 0:
    lstm = False
else:
    lstm = True

if cfg.attribute_depth == "depth":
    depth = True
else:
    depth = False

model = PConvLSTM(
    img_size=cfg.image_size,
    enc_dec_layers=cfg.encoding_layers,
    pool_layers=cfg.pooling_layers,
    in_channels=cfg.in_channels,
    out_channels=cfg.out_channels,
).to(cfg.device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
)

print(cfg.save_part)

model.eval()
prepo = preprocessing(
    cfg.im_dir,
    cfg.im_name,
    cfg.eval_im_year,
    cfg.image_size,
    "image",
    cfg.in_channels,
)
depths = prepo.depths()
argo = cfg.mask_argo
print(argo)

if cfg.combine_layers:
    evalu.combine_layers(
        np.arange(cfg.combine_start, cfg.combine_start + cfg.in_channels)
    )

else:
    start_iter = load_ckpt(
        "{}/ckpt/{}/{}.pth".format(cfg.save_dir, cfg.save_part, cfg.resume_iter),
        [("model", model)],
        cfg.device,
        [("optimizer", optimizer)],
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = cfg.lr
    val_dataset = MaskDataset(cfg.eval_im_year, cfg.in_channels, "eval", shuffle=False)
    evalu.infill(
        model,
        val_dataset,
        partitions=cfg.batch_size,
        iter=str(cfg.resume_iter),
        name=f"assimilation_{argo}",
    )
    val_obs_dataset = ValDataset(
        cfg.eval_im_year, cfg.mask_year, depth, cfg.in_channels
    )
    evalu.infill(
        model,
        val_obs_dataset,
        partitions=cfg.batch_size,
        iter=str(cfg.resume_iter),
        name=f"observations_{argo}",
    )

if cfg.eval_full:
    # evaluate assimilation reconstruction
    evalu.area_cutting(mode=f"assimilation_{argo}", depth=cfg.in_channels)
    evalu.heat_content(depths, str(cfg.resume_iter), name=f"assimilation_{argo}")
    evalu.pattern_corr_timeseries(name=f"assimilation_{argo}", del_t=12)
    evalu.pattern_corr_timeseries(name=f"assimilation_{argo}", del_t=1)
    evalu.pattern_corr_timeseries(name=f"assimilation_{argo}", del_t=5 * 12)
    evalu.heat_content_timeseries(
        depths, str(cfg.resume_iter), name=f"assimilation_{argo}"
    )

    # evaluate observations reconstruction
    evalu.area_cutting(mode=f"observations_{argo}", depth=cfg.in_channels)
    evalu.heat_content(depths, str(cfg.resume_iter), name=f"observations_{argo}")
    evalu.pattern_corr_timeseries(name=f"observations_{argo}", del_t=12)
    evalu.pattern_corr_timeseries(name=f"observations_{argo}", del_t=1)
    evalu.pattern_corr_timeseries(name=f"observations_{argo}", del_t=5 * 12)
    evalu.heat_content_timeseries(
        depths, str(cfg.resume_iter), name=f"observations_{argo}"
    )

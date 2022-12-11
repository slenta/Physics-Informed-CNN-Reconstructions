# script to evaluate error with iterations


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
import h5py
import os

matplotlib.use("Agg")

if not os.path.exists(f"{cfg.save_dir}/images/{cfg.save_part}/"):
    os.makedirs(f"{cfg.save_dir}/images/{cfg.save_part}/")
if not os.path.exists(f"{cfg.save_dir}/ckpt/{cfg.save_part}/"):
    os.makedirs(f"{cfg.save_dir}/ckpt/{cfg.save_part}/")
if not os.path.exists(f"{cfg.save_dir}/validation/{cfg.save_part}/"):
    os.makedirs(f"{cfg.save_dir}/validation/{cfg.save_part}/")

cfg.set_train_args()

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
argo = cfg.mask_argo

stopping_points = np.arange(
    cfg.save_model_interval, cfg.resume_iter, cfg.save_model_interval
)
rsmes = []

for iter in stopping_points:
    print(iter)
    start_iter = load_ckpt(
        "{}/ckpt/{}/{}.pth".format(cfg.save_dir, cfg.save_part, iter),
        [("model", model)],
        cfg.device,
        [("optimizer", optimizer)],
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = cfg.lr
    val_dataset = MaskDataset(cfg.eval_im_year, cfg.in_channels, "eval", shuffle=False)
    output, gt = evalu.infill(
        model,
        val_dataset,
        partitions=cfg.batch_size,
        iter=str(iter),
        name=f"assimilation_{argo}",
    )
    output = np.nanmean(output)
    gt = np.nanmean(gt)

    rsme = np.sqrt(output - gt) ** 2

    rsmes.append(rsme)

rsmes = np.array(rsmes)

file = f"{cfg.val_dir}{cfg.save_part}/val_errors.hdf5"
f = h5py.File(file, "w")
f.create_dataset(name="rsmes", shape=rsmes.shape, data=rsmes)
f.close()

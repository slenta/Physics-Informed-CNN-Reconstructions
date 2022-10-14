#Script to infill and evaluate certain model version


import matplotlib
import evaluation_og as evalu
from dataloader import MaskDataset
from dataloader import ValDataset
from preprocessing import preprocessing
from utils.io import load_ckpt, save_ckpt
import config as cfg
import torch
from model.net import PConvLSTM

matplotlib.use('Agg')

cfg.set_train_args()

if cfg.lstm_steps == 0:
        lstm = False
else:
        lstm = True

if cfg.attribute_depth == 'depth':
        depth = True
else:
        depth = False

model = PConvLSTM(img_size=cfg.image_size,
                  enc_dec_layers=cfg.encoding_layers,
                  pool_layers=cfg.pooling_layers,
                  in_channels=cfg.in_channels,
                  out_channels=cfg.out_channels).to(cfg.device)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)

print(cfg.save_part)

start_iter = load_ckpt(
        '{}/ckpt/{}/{}.pth'.format(cfg.save_dir, cfg.save_part, cfg.resume_iter), [('model', model)], cfg.device, [('optimizer', optimizer)])

for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.lr

model.eval()
prepo = preprocessing(cfg.im_dir, cfg.im_name, cfg.eval_im_year, cfg.image_size, 'image', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
depths = prepo.depths()


val_dataset = MaskDataset(cfg.eval_im_year, cfg.in_channels, 'eval', shuffle=False)
evalu.infill(model, val_dataset, partitions = cfg.batch_size, iter=str(cfg.resume_iter), name='assimilation_full')
evalu.h5_to_netcdf_cutting(mode='assimilation_full', depth=cfg.in_channels)
evalu.pattern_corr_timeseries(name='assimilation_full')
evalu.heat_content_correlation(depths, str(cfg.resume_iter), name='assimilation_full')
evalu.heat_content_timeseries(depths, str(cfg.resume_iter), name='assimilation_full')

val_obs_dataset = ValDataset(cfg.eval_im_year, cfg.eval_mask_year, depth, cfg.in_channels)
evalu.infill(model, val_obs_dataset, partitions=cfg.batch_size, iter=str(cfg.resume_iter), name='observations_full')
evalu.h5_to_netcdf_cutting(mode='observations_full', depth=cfg.in_channels)
evalu.pattern_corr_timeseries(name='observations_full')
evalu.heat_content_correlation(depths, str(cfg.resume_iter), name='observations_full')
evalu.heat_content_timeseries(depths, str(cfg.resume_iter), name='observations_full')
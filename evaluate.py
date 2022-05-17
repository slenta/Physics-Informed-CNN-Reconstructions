#Script to infill and evaluate certain model version


import evaluation_og as evalu
from dataloader import MaskDataset
from dataloader import ValDataset
from preprocessing import preprocessing
from utils.io import load_ckpt, save_ckpt
import config as cfg
import torch
from model.net import PConvLSTM


cfg.set_train_args()

lstm = False
depth=True

model = PConvLSTM(radar_img_size=cfg.image_size,
                  radar_enc_dec_layers=cfg.encoding_layers[0],
                  radar_pool_layers=cfg.pooling_layers[0],
                  radar_in_channels=cfg.in_channels,
                  radar_out_channels=cfg.out_channels,
                  lstm=lstm).to(cfg.device)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)



start_iter = load_ckpt(
        '{}/ckpt/{}/{}.pth'.format(cfg.save_dir, cfg.save_part, cfg.resume_iter), [('model', model)], cfg.device, [('optimizer', optimizer)])

for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.lr

model.eval()
prepo = preprocessing(cfg.im_dir, cfg.im_name, cfg.eval_im_year, cfg.image_size, 'image', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
depths = prepo.depths()
#prepo_obs = preprocessing(cfg.mask_dir, cfg.mask_name, cfg.eval_mask_year, cfg.image_size, 'val', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
#prepo_obs.save_data()
#prepo.save_data()

print('start')

val_dataset = MaskDataset(cfg.eval_im_year, depth, cfg.in_channels, 'eval', shuffle=False)
evalu.infill(model, val_dataset, partitions = cfg.batch_size, iter= str(cfg.resume_iter), name='assimilation_full')
evalu.heat_content_timeseries(depths, str(cfg.resume_iter), name='assimilation_full')

print('obs')

val_obs_dataset = ValDataset(cfg.eval_im_year, cfg.eval_mask_year, depth, cfg.in_channels)
evalu.infill(model, val_obs_dataset, partitions=cfg.batch_size, iter=str(cfg.resume_iter), name='observations_full')
evalu.heat_content_timeseries(depths, str(cfg.resume_iter), name='observations_full')
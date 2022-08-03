import os
import matplotlib
from sklearn.utils import shuffle
import torch
import sys

sys.path.append('./')

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.net import PConvLSTM
from utils.featurizer import VGG16FeatureExtractor
from utils.io import load_ckpt, save_ckpt
from utils.netcdfloader import InfiniteSampler
#from utils.evaluation import create_snapshot_image
from model.loss import InpaintingLoss, HoleLoss
import config as cfg
from dataloader import MaskDataset
import evaluation_og as evalu
from preprocessing import preprocessing
from dataloader import ValDataset
import time

torch.cuda.empty_cache()

matplotlib.use('Agg')



cfg.set_train_args()

if not os.path.exists(cfg.save_dir):
    os.makedirs('{:s}/{:s}/images'.format(cfg.save_dir, cfg.save_part))
    os.makedirs('{:s}/{:s}/ckpt'.format(cfg.save_dir, cfg.save_part))


log_dir = '{}{}/'.format(cfg.log_dir, cfg.save_part)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# create data sets
#dataset_train = NetCDFLoader(cfg.data_root_dir, cfg.img_names, cfg.mask_dir, cfg.mask_names, 'train', cfg.data_types,
#                             cfg.lstm_steps, cfg.prev_next_steps)
#dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.img_names, cfg.mask_dir, cfg.mask_names, 'val', cfg.data_types,
#                           cfg.lstm_steps, cfg.prev_next_steps)
if cfg.attribute_depth == 'depth':
    depth = True
else:
    depth = False

dataset_train = MaskDataset(cfg.im_year, depth, cfg.in_channels, mode='train')
dataset_test = MaskDataset(cfg.eval_im_year, depth, cfg.in_channels, mode='test')

iterator_train = iter(DataLoader(dataset_train, batch_size=cfg.batch_size,
                                 sampler=InfiniteSampler(len(dataset_train)),
                                 num_workers=cfg.n_threads))

# define network model
lstm = True
if cfg.lstm_steps == 0:
    lstm = False



before = time.time()

model = PConvLSTM(radar_img_size=cfg.image_size,
                  radar_enc_dec_layers=cfg.encoding_layers[0],
                  radar_pool_layers=cfg.pooling_layers[0],
                  radar_in_channels=cfg.in_channels,
                  radar_out_channels=cfg.out_channels,
                  lstm=lstm).to(cfg.device)

# define learning rate
if cfg.finetune:
    lr = cfg.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = cfg.lr


# define optimizer and loss functions
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
if cfg.loss_criterion == 1:
    criterion = HoleLoss().to(cfg.device)
    lambda_dict = cfg.LAMBDA_DICT_HOLE
else:
    criterion = InpaintingLoss(VGG16FeatureExtractor()).to(cfg.device)
    lambda_dict = cfg.LAMBDA_DICT_IMG_INPAINTING


# define start point
start_iter = 0
if cfg.resume_iter:
    start_iter = load_ckpt(
        '{}ckpt/{}/{}.pth'.format(cfg.save_dir, cfg.save_part, cfg.resume_iter), [('model', model)], cfg.device, [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

for i in tqdm(range(start_iter, cfg.max_iter)):

    start = time.time()


    # train model
    model.train()
    image, mask, gt, im_rea, mask_rea = [x.to(cfg.device) for x in next(iterator_train)]
    output = model(image, mask, im_rea, mask_rea)

    # calculate loss function and apply backpropagation
    loss_dict = criterion(mask[:, :, :, :],
                          output[:, :, :, :],
                          gt[:, :, :, :])
    loss = 0.0
    for key, factor in lambda_dict.items():
        value = factor * loss_dict[key]
        loss += value
        if cfg.log_interval and (i + 1) % cfg.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # save checkpoint
    if (i + 1) % cfg.save_model_interval == 0 or (i + 1) == cfg.max_iter:
        save_ckpt('{:s}ckpt/{:s}/{:d}.pth'.format(cfg.save_dir, cfg.save_part, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)

    # create snapshot image
    if (i + 1) % cfg.vis_interval == 0:
        model.eval()
        evalu.evaluate(model, dataset_test, cfg.device,
                 '{:s}/images/{:s}/test_{:d}'.format(cfg.save_dir, cfg.save_part, i + 1))

    #validate using validation ensemble member and create ohc timeseries
    if (i + 1) % cfg.val_interval == 0:
        model.eval()
        prepo = preprocessing(cfg.im_dir, cfg.im_name, cfg.eval_im_year, cfg.image_size, 'image', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
        #prepo_obs = preprocessing(cfg.mask_dir, cfg.mask_name, cfg.eval_mask_year, cfg.image_size, 'val', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
        #prepo_obs.save_data()
        #prepo.save_data()
        depths = prepo.depths()

        val_dataset = MaskDataset(cfg.eval_im_year, depth, cfg.in_channels, 'eval', shuffle=False)
        evalu.infill(model, val_dataset, partitions = cfg.batch_size, iter= str(i+1), name='_assimilation')
        evalu.heat_content_timeseries(depths, str(i+1), name='_assimilation')

        #val_obs_dataset = ValDataset(cfg.eval_im_year, cfg.eval_mask_year, depth, cfg.in_channels)
        #evalu.infill(model, val_obs_dataset, partitions=cfg.batch_size, iter=str(i + 1), name='_observations')
        #evalu.heat_content_timeseries(depths, str(i + 1), name='_observations')


    #if cfg.save_snapshot_image and (i + 1) % cfg.log_interval == 0:
    #    model.eval()
    #    create_snapshot_image(model, dataset_val, '{:s}/images/Maske_{:d}/iter_{:f}'.format(cfg.snapshot_dir, cfg.mask_year, i + 1))

    #print(f'Time: {start - before}, {start - third}, {second - start}, {third - second}')


writer.close()

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

if not os.path.exists(f'{cfg.save_dir}/images/{cfg.save_part}/'):
    os.makedirs(f'{cfg.save_dir}/images/{cfg.save_part}/')
if not os.path.exists(f'{cfg.save_dir}/ckpt/{cfg.save_part}/'):
    os.makedirs(f'{cfg.save_dir}/ckpt/{cfg.save_part}/')
if not os.path.exists(f'{cfg.save_dir}/validation/{cfg.save_part}/'):
    os.makedirs(f'{cfg.save_dir}/validation/{cfg.save_part}/') 


log_dir = f'{cfg.log_dir}{cfg.save_part}/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# define lstm, depth variables
if cfg.attribute_depth == 'depth':
    depth = True
else:
    depth = False

# define datasets
dataset_train = MaskDataset(cfg.im_year, depth, cfg.in_channels, mode='train')
dataset_test = MaskDataset(cfg.eval_im_year, depth, cfg.in_channels, mode='test')

iterator_train = iter(DataLoader(dataset_train, batch_size=cfg.batch_size,
                                 sampler=InfiniteSampler(len(dataset_train)),
                                 num_workers=cfg.n_threads))

#define network model
model = PConvLSTM(img_size=cfg.image_size,
                  enc_dec_layers=cfg.encoding_layers[0],
                  pool_layers=cfg.pooling_layers[0],
                  in_channels=cfg.in_channels,
                  out_channels=cfg.out_channels).to(cfg.device)

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

    # train model
    model.train()
    image, mask, gt = [x.to(cfg.device) for x in next(iterator_train)]
    output = model(image, mask)

    # calculate loss function and apply backpropagation
    if cfg.lstm_steps != 0:
        loss_dict = criterion(mask[:, cfg.lstm_steps - 1, :, :, :],
                              output[:, cfg.lstm_steps - 1, :, :, :],
                              gt[:, cfg.lstm_steps - 1, :, :, :])
    else:
        loss_dict = criterion(mask, output, gt)    

    
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
        evalu.create_snapshot_image(model, dataset_test, f'{cfg.save_dir}/images/{cfg.save_part}/iter_{str(i + 1)}')

    #validate using test dataset 
    if (i + 1) % cfg.val_interval == 0:
        model.eval()
        evalu.evaluate(model, dataset_test, cfg.device,
                 '{:s}/images/{:s}/test_{:d}'.format(cfg.save_dir, cfg.save_part, i + 1))

writer.close()

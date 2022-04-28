import config as cfg
from dataloader import MaskDataset
from utils.netcdfloader import InfiniteSampler
from torch.utils.data import DataLoader
from preprocessing import preprocessing

cfg.set_train_args()


if cfg.depth:
    depth = True
else:
    depth = False

dataset_train = MaskDataset(cfg.im_year, depth, cfg.in_channels, mode='train')
dataset_test = MaskDataset(cfg.im_year, depth, cfg.in_channels, mode='test')

iterator_train = iter(DataLoader(dataset_train, batch_size=cfg.batch_size,
                                 sampler=InfiniteSampler(len(dataset_train)),
                                 num_workers=cfg.n_threads))



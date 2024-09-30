import torch
import config as cfg


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(cfg.STD) + torch.Tensor(cfg.MEAN)
    x = x.transpose(1, 3)
    return x

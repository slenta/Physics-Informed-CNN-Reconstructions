from base64 import decode
import config as cfg
from dataloader import MaskDataset
from utils.netcdfloader import InfiniteSampler
import h5py
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

a, b, c, d = np.zeros((4,8, 8))

print(d)
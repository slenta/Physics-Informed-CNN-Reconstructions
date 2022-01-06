import h5py
from matplotlib import image
import numpy as np
import pylab as plt
import torch
from dataloader import MaskDataset




def visualisation(iter):
    
    f = h5py.File('../Asi_maskiert/results/images/test_' + iter + '.hdf5', 'r')
    fm = h5py.File('../Asi_maskiert/original_image/Assimilation_1958_2020.hdf5', 'r')
    original = fm.get('tos_sym')[1]
    image_data = f.get('image')[2, :, :]
    #mask_data = f.get('mask')[2, :, :]
    output_data = f.get('output')[2, :, :]
    output_comp = f.get('output_comp')[2, :, :]
    #img, msk, gt = MaskDataset('2020')[0]

    #mask = torch.from_numpy(mask_data)
    output = torch.from_numpy(output_data)
    image = torch.from_numpy(image_data)

    print(output)

    plt.figure(figsize=(24, 6))
    plt.subplot(1, 4, 1)
    plt.title('Masked Image')
    im1 = plt.imshow(image_data, vmin=0, vmax=40, cmap='jet', aspect='auto')
    plt.subplot(1, 4, 2)
    plt.title('NN Output')
    im2 = plt.imshow(output, cmap = 'jet', vmin=-10, vmax=40, aspect = 'auto')
    plt.subplot(1, 4, 3)
    plt.title('Original Assimilation Image')
    im3 = plt.imshow(original, cmap='jet', vmin=-10, vmax=40, aspect='auto')
    plt.subplot(1, 4, 4)
    plt.title('Output Composition')
    im5 = plt.imshow(output_comp, vmin=-10, vmax=40, cmap='jet', aspect='auto')
    plt.colorbar(im2)
    plt.savefig('../Asi_maskiert/results/images/part_1/test_' + iter + '.pdf')
    plt.show()

visualisation('200000')

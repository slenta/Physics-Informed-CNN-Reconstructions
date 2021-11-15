import h5py
import numpy as np
import pylab as plt




def visualisation(iter):
    
    f = h5py.File('../Asi_maskiert/results/images/test_' + iter + '.hdf5', 'r')
    image_data = f.get('image')[7, 2, :, :]
    #mask_data = f.get('mask')[7, 2, :, :]
    output_data = f.get('output')[7, 2, :, :]

    plt.subplot(1, 2, 1)
    plt.imshow(image_data, vmin=-10, vmax=40, cmap='jet', aspect='auto')
    #plt.subplot(1, 2, 2)
    #plt.imshow(mask_data, vmin=0, vmax=10, cmap='jet', aspect='auto')
    plt.subplot(1, 2, 2)
    plt.imshow(output_data, vmin=-10, vmax=40, cmap='jet', aspect='auto')
    plt.colorbar()
    plt.show()

visualisation('1')

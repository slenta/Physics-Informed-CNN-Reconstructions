import h5py
import numpy as np
import pylab as plt




def visualisation(iter):
    
    f = h5py.File('../Asi_maskiert/results/images/test_' + iter + '.hdf5', 'r')
    image_data = f.get('image')[7, 2, :, :]
    #mask_data = f.get('mask')[7, 2, :, :]
    output_data = f.get('output')[7, 2, :, :]
    output_comp = f.get('output_comp')[7, 2, :, :]

    plt.subplot(1, 3, 1)
    plt.imshow(image_data, vmin=-10, vmax=40, cmap='jet', aspect='auto')
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(output_comp, cmap='jet', aspect='auto')
    plt.subplot(1, 3, 3)
    im1 = plt.imshow(output_data, cmap='jet', vmin = 0, vmax = 0.1, aspect='auto')
    #print(output_data)
    plt.colorbar(im2)
    plt.show()
    #print(output_data, output_comp)


visualisation('2')

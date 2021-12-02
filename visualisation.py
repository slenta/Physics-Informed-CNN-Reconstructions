from typing import final
import h5py
import numpy as np
import pylab as plt
import torch




def visualisation(iter):
    
    f = h5py.File('../Asi_maskiert/results/images/test_' + iter + '.hdf5', 'r')
    fm = h5py.File('../Asi_maskiert/original_image/Assimilation_1958_2020.hdf5')
    original = fm.get('tos_sym')[1]
    image_data = f.get('image')[7, 2, :, :]
    #mask_data = f.get('mask')[7, 2, :, :]
    output_data = f.get('output')[7, 2, :, :]
    output_comp = f.get('output_comp')[7, 2, :, :]

    plt.figure(figsize=(14, 8))
    plt.subplot(1, 4, 1)
    plt.title('Masked Image')
    plt.imshow(image_data, vmin=-10, vmax=40, cmap='jet', aspect='auto')
    plt.subplot(1, 4, 2)
    plt.title('NN Output')
    im2 = plt.imshow(output_data, vmin=-0, vmax=0.01, cmap='jet', aspect='auto')
    plt.subplot(1, 4, 3)
    plt.title('Original Assimilation Image')
    im3 = plt.imshow(original, cmap='jet', vmin=-10, vmax=50, aspect='auto')
    plt.subplot(1, 4, 4)
    plt.title('Output Composition')
    im4 = plt.imshow(output_comp, cmap='jet', aspect='auto')
    #print(output_data)
    plt.colorbar(im4)
    #plt.savefig('../Asi_maskiert/pdfs/results/Erstes_Ergebnis_' + iter + '.pdf')
    plt.show()
    #print(output_data, output_comp)


visualisation('3')

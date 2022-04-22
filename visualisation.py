from cProfile import label
import h5py
from isort import file
from matplotlib import image
from matplotlib.pyplot import title
import numpy as np
import pylab as plt
import torch
import xarray as xr
from mpl_toolkits.mplot3d import axes3d
import cdo
cdo = cdo.Cdo()

def vis_single(timestep, path, name, argo_state, type, param, title):

    if type=='output':
        f = h5py.File(path + name + '.hdf5', 'r')

        output = f.get('output')[timestep, 0, :, :]
        image = f.get('image')[timestep, 0, :, :]
        mask = f.get('mask')[timestep, 0, :, :]
        masked = mask * image
        outputcomp = mask*image + (1 - mask)*output
        if param == 'mask':
                
            plt.figure(figsize=(6, 6))
            plt.title(title)
            #plt.imshow(sst, vmin=-5, vmax=30, cmap='jet')
            plt.imshow(masked, vmin = -5, vmax = 30, cmap='jet')
            plt.colorbar(label='Temperature in °C')
            plt.savefig('../Asi_maskiert/pdfs/' + name + param + argo_state + '.pdf')
            plt.show()
        elif param == 'image':
                           
            plt.figure(figsize=(6, 6))
            plt.title(title)
            #plt.imshow(sst, vmin=-5, vmax=30, cmap='jet')
            plt.imshow(outputcomp, vmin = -5, vmax = 30, cmap='jet')
            plt.colorbar(label='Temperature in °C')
            plt.savefig('../Asi_maskiert/pdfs/' + name + param + argo_state + '.pdf')
            plt.show()

    elif type=='image':
        df = xr.load_dataset(path + name + '.nc', decode_times=False)

        sst = df.thetao.values
        sst = sst[0, 10, :, :]
        x = np.isnan(sst)
        sst[x] = -15
        
        plt.figure(figsize=(6, 4))
        plt.title(title)
        #plt.imshow(sst, vmin=-5, vmax=30, cmap='jet')
        plt.imshow(sst, vmin = -5, vmax = 30)
        plt.colorbar(label='Temperature in °C')
        plt.savefig('../Asi_maskiert/pdfs/' + name + argo_state + '.pdf')
        plt.show()
    
    elif type=='mask':
        df = xr.load_dataset(path + name + '.nc', decode_times=False)
        sst = df.tho.values
        sst = sst[timestep, 10, :, :]

        x = np.isnan(sst)
        sst[x] = -15

        plt.figure(figsize=(8, 6))
        plt.title(title)
        plt.imshow(sst, vmin=-5, vmax=30, cmap='jet')
        plt.colorbar(label='Temperatur in °C')
        plt.savefig('../Asi_maskiert/pdfs/' + name + type + argo_state + '.pdf')
        plt.show()

    elif type=='3d':
        df = xr.load_dataset(path + name + '.nc', decode_times=False)

        sst = df.thetao.values[timestep, :, :, :]
        plt.figure()
        plt.title(title)
        ax = plt.subplot(111, projection='3d')

        z = sst[0, :, :]
        x = df.x.values
        y = df.y.values

        x = np.concatenate((np.zeros(17), x))
        print(x.shape, y.shape)
        scatter = ax.scatter(x, y, z, c=z, alpha=1)
        plt.colorbar(scatter, label='Temperatur in °C')

        plt.show()




def visualisation(path, iter, depth):
    
    f = h5py.File(path, 'r')
    fm = h5py.File('../Asi_maskiert/original_masks/Kontinentmaske.hdf5', 'r')
    
    continent_mask = fm.get('tos_sym')
    image_data = f.get('image')[1, depth, :, :]
    mask_data = f.get('mask')[1, depth,:, :]
    output_data = f.get('output')[1, depth,:, :]

    image_runter = f.get('image')[1, depth, :, :]
    
    print(np.sum(image_data - image_runter))

    error = np.zeros((8, 3))
    for i in range(8):
        for j in range(3):
            image = f.get('image')[i, j, :, :]
            output = f.get('output')[i, j, :, :]
            mask = f.get('mask')[i, j, :, :]
            outputcomp = mask*image + (1 - mask)*output

            error[i, j] = np.mean((np.array(outputcomp) - np.array(image))**2)


    print(error)

    mask = torch.from_numpy(mask_data)
    output = torch.from_numpy(output_data)
    image = torch.from_numpy(image_data)
    outputcomp = mask*image + (1 - mask)*output
    print(np.mean(error))

    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    fig.suptitle('Anomaly North Atlantic SSTs')
    plt.subplot(2, 2, 3)
    plt.title('Masked Image')
    im1 = plt.imshow(image * mask, cmap='jet', vmin=-3, vmax=3, aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.colorbar(label='Temperature in °C')
    plt.subplot(2, 2, 2)
    plt.title('NN Output')
    im2 = plt.imshow(outputcomp, cmap = 'jet', vmin=-3, vmax=3, aspect = 'auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.colorbar(label='Temperature in °C')
    plt.subplot(2, 2, 1)
    plt.title('Original Assimilation Image')
    im3 = plt.imshow(image, cmap='jet', vmin=-3, vmax=3, aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.colorbar(label='Temperature in °C')
    plt.subplot(2, 2, 4)
    plt.title('Error')
    im5 = plt.imshow(image - output, vmin=-1.5, vmax=1.5, cmap='jet', aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.colorbar(label='Temperature in °C')
    #plt.savefig('../Asi_maskiert/results/images/depth/test_' + iter + '.pdf')
    plt.show()

visualisation('../Asi_maskiert/results/images/depth_10/test_' + '50000' + '.hdf5', '50000', 9)

#vis_single(753, '../Asi_maskiert/original_image/', 'Image_3d_newgrid', 'r1011_shuffle_newgrid/short_val/Maske_1970_1985r1011_shuffle_newgrid/short_val/Maske_1970_1985Argo-era', 'image', 'image', 'North Atlantic Assimilation October 2020')
#vis_single(9, '../Asi_maskiert/original_masks/', 'Maske_2020_newgrid', 'pre-Argo-era', 'mask', 'mask', 'North Atlantic Observations October 2020')

#vis_single(1, '../Asi_maskiert/results/images/r1011_shuffle_newgrid/short_val/Maske_1970_1985/', 'test_700000', 'pre-argo-era', 'output', 'mask', 'Pre-Argo-Era Masks')
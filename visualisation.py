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
import config as cfg

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

def vis_variable(path_1, path_2, path_3, var_1, var_2, var_3, name):

    f1 = h5py.File(path_1, 'r')
    f2 = h5py.File(path_2, 'r')
    f3 = h5py.File(path_3, 'r')

    v1 = f1.get(var_1)
    v2 = f2.get(var_2)
    v3 = f3.get(var_3)

    fc = h5py.File('../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5', 'r')
    continent_mask = fc.get('tos_sym')


    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    fig.suptitle('Anomaly North Atlantic SSTs')
    plt.subplot(1, 3, 1)
    plt.title('Step 1: Binary Assimilation Mask: October 2020')
    current_cmap = plt.cm.jet
    current_cmap.set_bad(color='gray')
    im1 = plt.imshow(v1, cmap=current_cmap, vmin=-3, vmax=3, aspect='auto', interpolation=None)
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    #plt.colorbar(label='Temperature in °C')
    plt.subplot(1, 3, 2)
    plt.title('Step 2: Binary Assimilation Mask: January 1970')
    im2 = plt.imshow(v2, cmap = 'jet', vmin=-3, vmax=3, aspect = 'auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    #plt.colorbar(label='Temperature in °C')
    plt.subplot(1, 3, 3)
    plt.title('Step 3: Observations: January 1970')
    im3 = plt.imshow(v3, cmap='jet', vmin=-3, vmax=3, aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.colorbar(label='Temperature in °C')
    fig.savefig('../Asi_maskiert/pdfs/' + name + '.pdf', dpi = fig.dpi)
    plt.show()


def visualisation(path, iter, depth):
    
    f = h5py.File(path + iter + '.hdf5', 'r')
    fm = h5py.File('../Asi_maskiert/original_masks/Kontinent_newgrid.hdf5', 'r')
    
    continent_mask = fm.get('tos_sym')
    image_data = f.get('gt')[0, depth, :, :]
    mask_data = f.get('mask')[0, depth,:, :]
    output_data = f.get('output')[0, depth,:, :]
    continent_mask = np.array(continent_mask)



    error = np.zeros((8, depth))
    for i in range(8):
        for j in range(depth):
            image = f.get('image')[i, j, :, :]
            output = f.get('output')[i, j, :, :]
            mask = f.get('mask')[i, j, :, :]
            outputcomp = mask*image + (1 - mask)*output

            error[i, j] = np.mean((np.array(outputcomp) - np.array(image))**2)


    print(error)

    n = mask_data.shape
    mask_grey = np.zeros(n)

    for i in range(n[0]):
        for j in range(n[1]):
            if mask_data[i, j] == 0:
                mask_grey[i, j] = np.NaN
            else:
                mask_grey[i, j] = mask_data[i, j]

    continent_mask = torch.from_numpy(continent_mask)
    mask_grey = torch.from_numpy(mask_grey) * continent_mask
    mask = torch.from_numpy(mask_data) * continent_mask
    output = torch.from_numpy(output_data) * continent_mask
    image = torch.from_numpy(image_data) * continent_mask
    outputcomp = mask*image + (1 - mask)*output
    print(np.mean(error))

    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    fig.suptitle('Anomaly North Atlantic SSTs')
    plt.subplot(1, 3, 1)
    plt.title('Masked Image')
    current_cmap = plt.cm.jet
    current_cmap.set_bad(color='gray')
    im1 = plt.imshow(image * mask_grey, cmap=current_cmap, vmin=-3, vmax=3, aspect='auto', interpolation=None)
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    #plt.colorbar(label='Temperature in °C')
    plt.subplot(1, 3, 2)
    plt.title('NN Output')
    im2 = plt.imshow(outputcomp, cmap = 'jet', vmin=-3, vmax=3, aspect = 'auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    #plt.colorbar(label='Temperature in °C')
    plt.subplot(1, 3, 3)
    plt.title('Original Assimilation Image')
    im3 = plt.imshow(image, cmap='jet', vmin=-3, vmax=3, aspect='auto')
    plt.xlabel('Transformed Longitudes')
    plt.ylabel('Transformed Latitudes')
    plt.colorbar(label='Temperature in °C')
    #plt.subplot(2, 2, 4)
    #plt.title('Error')
    #im5 = plt.imshow(image - output, vmin=-1.5, vmax=1.5, cmap='jet', aspect='auto')
    #plt.xlabel('Transformed Longitudes')
    #plt.ylabel('Transformed Latitudes')
    #plt.colorbar(label='Temperature in °C')
    fig.savefig(path + iter + '.pdf', dpi = fig.dpi)
    plt.show()


                
def timeseries_plotting(path, iteration):
    f = h5py.File(cfg.val_dir + path + 'timeseries_' + str(iteration) + '.hdf5', 'r')
    f1_compare = h5py.File(cfg.val_dir + 'validation_timeseries_r12_newgrid.hdf5', 'r')
    f2_compare = h5py.File(cfg.val_dir + 'validation_timeseries_r13_newgrid.hdf5', 'r')
    f3_compare = h5py.File(cfg.val_dir + 'validation_timeseries_r14_newgrid.hdf5', 'r')
    fo = h5py.File(cfg.val_dir + 'part_2/timeseries_50001.hdf5', 'r')
 


    hc_c1 = f1_compare.get('gt_ts')
    hc_c2 = f2_compare.get('gt_ts')
    hc_c3 = f3_compare.get('gt_ts')
    hc_network = f.get('network_ts')
    hc_gt = f.get('gt_ts')
    t_mean = f1_compare.get('mean_temp')

    hc_network = np.array(hc_network)
    hc_gt = np.array(hc_gt)
    hc_c1 = np.array(hc_c1)
    hc_c2 = np.array(hc_c2)
    hc_c3 = np.array(hc_c3)
    tm = np.array(t_mean)
    hc_obs = np.array(fo.get('network_ts'))

    #f_og = h5py.File(cfg.val_dir + str(iteration) + '.hdf5', 'r')
    #output_comp = f_og.get('output_comp')

    plt.figure(figsize=(10, 6))

    plt.plot(hc_network, label='Network Reconstructed Heat Content')
    plt.plot(hc_gt, label='Assimilation Heat Content')
    #plt.plot(hc_c1, label='Comparison ensemble member', color='red')
    #plt.plot(tm, label='Comparison ensemble member', color='red')
    plt.plot(hc_c2, label='Comparison ensemble member', color='red')
    plt.plot(hc_c3, label='Comparison ensemble member', color='red')
    plt.plot(hc_obs, label='Observations reconstruction')
    plt.plot()
    plt.grid()
    plt.legend()
    plt.title('Comparison Reconstruction to Assimilation Timeseries')
    plt.xlabel('Months since January 2004')
    plt.ylabel('Heat Content [J/m²]')
    plt.savefig('../Asi_maskiert/pdfs/validation_timeseries_obs' + str(iteration) + '.pdf')
    plt.show()


cfg.set_train_args()
#visualisation('../Asi_maskiert/results/images/Maske_preargo_20/test_', '25000', 0)
timeseries_plotting('../Asi_maskiert/results/validation/timeseries_', 25000)



#vis_single(753, '../Asi_maskiert/original_image/', 'Image_3d_newgrid', 'r1011_shuffle_newgrid/short_val/Maske_1970_1985r1011_shuffle_newgrid/short_val/Maske_1970_1985Argo-era', 'image', 'image', 'North Atlantic Assimilation October 2020')
#vis_single(9, '../Asi_maskiert/original_masks/', 'Maske_2020_newgrid', 'pre-Argo-era', 'mask', 'mask', 'North Atlantic Observations October 2020')

#vis_single(1, '../Asi_maskiert/results/images/r1011_shuffle_newgrid/short_val/Maske_1970_1985/', 'test_700000', 'pre-argo-era', 'output', 'mask', 'Pre-Argo-Era Masks')
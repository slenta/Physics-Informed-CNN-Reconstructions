from numpy.core.fromnumeric import shape
import torch
from torch.utils import data
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
from image import unnormalize
import h5py
import matplotlib.pyplot as plt
import config as cfg
from numpy import ma
import sys
import os



sys.path.append('./')
cfg.set_train_args()

def evaluate(model, dataset, device, filename):
    image, mask, gt, i1, m1 = zip(*[dataset[i] for i in range(8)])

    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    i1 = torch.stack(i1)
    m1 = torch.stack(m1)

    image = torch.as_tensor(image)
    mask = torch.as_tensor(mask)
    gt = torch.as_tensor(gt)
    i1 = torch.as_tensor(i1)
    m1 = torch.as_tensor(m1)

    with torch.no_grad():
        output = model(image.to(device), mask.to(device), i1.to(device), m1.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask*image + (1 - mask)*output

    #grid = make_grid(
    #    torch.cat((unnormalize(image), unnormalize(mask), unnormalize(output),
    #               unnormalize(output_comp), unnormalize(gt)), dim=0))

    n = image.shape

    f = h5py.File(filename + '.hdf5', 'w')
    dset1 = f.create_dataset('image', (n[0], n[1], n[2], n[3]), dtype = 'float32',data = image)
    dset2 = f.create_dataset('output', (n[0], n[1], n[2], n[3]), dtype = 'float32',data = output)
    dset3 = f.create_dataset('output_comp', (n[0], n[1], n[2], n[3]), dtype = 'float32',data = output_comp)
    dset4 = f.create_dataset('mask', shape=(n[0], n[1], n[2], n[3]), dtype='float32', data=mask)
    dset5 = f.create_dataset('gt', (n[0], n[1], n[2], n[3]), dtype = 'float32',data = gt)
    f.close()
    
    #save_image(grid, filename + '.jpg')

def infill(model, dataset, partitions, iter, name):
    if not os.path.exists(cfg.val_dir):
        os.makedirs('{:s}'.format(cfg.val_dir))
    image = []
    mask = []
    gt = []
    output = []


    if partitions > dataset.__len__():
        partitions = dataset.__len__()
    
    print(dataset.__len__())
    if dataset.__len__() % partitions != 0:
        print("WARNING: The size of the dataset should be dividable by the number of partitions. The last "
              + str(dataset.__len__() % partitions) + " time steps will not be infilled.")
    for split in range(partitions):
        image_part, mask_part, gt_part, rea_images_part, rea_masks_part = zip(
            *[dataset[i + split * (dataset.__len__() // partitions)] for i in
              range(dataset.__len__() // partitions)])
        image_part = torch.stack(image_part)
        mask_part = torch.stack(mask_part)
        gt_part = torch.stack(gt_part)
        rea_images_part = torch.stack(rea_images_part)
        rea_masks_part = torch.stack(rea_masks_part)
        # get results from trained network
        with torch.no_grad():
            output_part = model(image_part.to(cfg.device), mask_part.to(cfg.device),
                                rea_images_part.to(cfg.device), rea_masks_part.to(cfg.device))

        if cfg.lstm_steps == 0:

            image_part = image_part[:, :, :, :].to(torch.device('cpu'))
            mask_part = mask_part[:, :, :, :].to(torch.device('cpu'))
            gt_part = gt_part[:, :, :, :].to(torch.device('cpu'))
            output_part = output_part[:, :, :, :].to(torch.device('cpu'))

        else:

            image_part = image_part[:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))
            mask_part = mask_part[:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))
            gt_part = gt_part[:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))
            output_part = output_part[:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))            


        image.append(image_part)
        mask.append(mask_part)
        gt.append(gt_part)
        output.append(output_part)

    image = torch.cat(image)
    mask = torch.cat(mask)
    gt = torch.cat(gt)
    output = torch.cat(output)

    # create output_comp
    output_comp = mask * image + (1 - mask) * output
    cvar = [gt, output, output_comp, image, mask]
    cname = ['gt', 'output', 'output_comp', 'image', 'mask']
    dname = ['time', 'lat', 'lon']
    
    h5 = h5py.File(cfg.val_dir + cfg.save_part + '/validation_'  + iter + '_' + name + '.hdf5', 'w')
    for x in range(0, 2):
        h5.create_dataset(name=cname[x], shape=cvar[x].shape, dtype=float, data=cvar[x].to(torch.device('cpu')))
        #for dim in range(0, 3):
        #    h5[cfg.data_type].dims[dim].label = dname[dim]
    h5.close()

    return ma.masked_array(gt, mask)[:, :, :, :], ma.masked_array(output_comp, mask)[:, :, :, :]



def heat_content_timeseries(depth_steps, iteration, name):

    rho = 1025  #density of seawater
    shc = 3850  #specific heat capacity of seawater

    f = h5py.File(cfg.val_dir + cfg.save_part + '/validation_' + iteration + '_' + name + '.hdf5', 'r')
    output = f.get('output')
    gt = f.get('gt')

    #take spatial mean of network output and ground truth
    output = np.mean(np.mean(output, axis=2), axis=2)
    gt = np.mean(np.mean(gt, axis=2), axis=2)
    n = output.shape
    hc_network = np.zeros(n[0])
    hc_assi = np.zeros(n[0])

    for i in range(n[0]):
        hc_network[i] = np.sum([(depth_steps[k] - depth_steps[k-1])*output[i, k]*rho*shc for k in range(1, n[1])]) + depth_steps[0] * output[i, 0] * rho * shc
        hc_assi[i] = np.sum([(depth_steps[k] - depth_steps[k-1])*gt[i, k]*rho*shc for k in range(1, n[1])]) + depth_steps[0] * gt[i, 0] * rho * shc


    f_final = h5py.File(cfg.val_dir + cfg.save_part + '/timeseries_' + iteration + '_' + name + '.hdf5', 'w')
    f_final.create_dataset(name='network_ts', shape=hc_network.shape, dtype=float, data=hc_network)
    f_final.create_dataset(name='gt_ts', shape=hc_assi.shape, dtype=float, data=hc_assi)
    f.close()

def heat_content_timeseries_masked(depth_steps, im_year, mask_year):

    rho = 1025  #density of seawater
    shc = 3850  #specific heat capacity of seawater

    f = h5py.File(cfg.im_dir + cfg.im_name + im_year + '_' +  cfg.attribute_depth + '_' + cfg.attribute_anomaly + '_full_' + str(cfg.in_channels) + '.hdf5', 'r')
    fo = h5py.File(cfg.mask_dir + cfg.mask_name + mask_year+ '_' +  cfg.attribute_depth + '_' + cfg.attribute_anomaly + '_full_' + str(cfg.in_channels) + '_observations.hdf5', 'r')
    
    image = f.get('tos_sym')
    obs = fo.get('tos_sym')
    obs = np.array(obs)

    obs = np.where(obs==0, np.nan, obs)
    obs_binary = np.where(np.isnan(obs)==False, 1, obs)
    
    plt.figure()
    plt.imshow(obs[0, 0, :, :])
    plt.savefig(cfg.save_dir + 'obs_binary.pdf')
    plt.show()

    image = image*obs_binary

    #take spatial mean of network output and ground truth
    image = np.nanmean(np.nanmean(image, axis=2), axis=2)
    obs = np.nanmean(np.nanmean(obs, axis=2), axis=2)
    
    n = image.shape
    hc_assi = np.zeros(n[0])
    hc_obs = np.zeros(n[0])

    for i in range(n[0]):
        hc_assi[i] = np.sum([(depth_steps[k] - depth_steps[k-1])*image[i, k]*rho*shc for k in range(1, n[1])]) + depth_steps[0] * image[i, 0] * rho * shc
        hc_obs[i] = np.sum([(depth_steps[k] - depth_steps[k-1])*obs[i, k]*rho*shc for k in range(1, n[1])]) + depth_steps[0] * obs[i, 0] * rho * shc


    f_final = h5py.File(cfg.val_dir + 'masked_timeseries_' + im_year + '.hdf5', 'w')
    f_final.create_dataset(name='im_ts', shape=hc_assi.shape, dtype=float, data=hc_assi)
    f_final.create_dataset(name='obs_ts', shape=hc_obs.shape, dtype=float, data=hc_obs)
    f.close()


def compare_datasets(obs_path, im_path, name):

    f1 = h5py.File(obs_path, 'r')
    f3 = h5py.File(im_path, 'r')

    obs = f1.get('tos_sym')
    image = f3.get('tos_sym')
    obs = np.array(obs)


    obs = np.where(obs==0, np.nan, obs)
    obs_binary = np.where(np.isnan(obs)==False, 1, obs)
    masked = obs_binary*image

    std_1 = np.nanstd(np.nanstd(obs, axis=0), axis=0)
    std_2 = np.nanstd(np.nanstd(masked, axis=0), axis=0)
    bias = np.nanmean(np.nanmean(obs, axis=0), axis=0) - np.nanmean(np.nanmean(masked, axis=0), axis=0)
    std_diff = std_1 - std_2


    
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.title('Standard Deviation Observations')
    plt.imshow(std_1, vmin = -1, vmax=1, cmap='coolwarm')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.title('Standard deviation Assmilation')
    plt.imshow(std_2, vmin=-1, vmax=1, cmap='coolwarm')
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(std_diff, vmin=-1, vmax=1, cmap='coolwarm')
    plt.colorbar()
    plt.title('Difference: Obs - Assimilation')
    plt.subplot(2, 2, 4)
    plt.imshow(bias, vmin=-1, vmax=1, cmap='coolwarm')
    plt.colorbar()
    plt.title('Bias: Obs - Assimilation')
    plt.savefig(cfg.save_dir + 'Dataset_diff' + name + '.pdf')
    plt.show()






#cfg.set_train_args()

#prepo = preprocessing(cfg.im_dir, cfg.im_name, cfg.eval_im_year, cfg.image_size, 'image', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
#prepo.save_data()
#depths = prepo.depths()

#heat_content_timeseries_general(depths, cfg.eval_im_year)


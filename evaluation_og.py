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
from utils.corr_2d_ttest import corr_2d_ttest
from collections import namedtuple


sys.path.append('./')
cfg.set_train_args()

def evaluate(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(8)])

    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)

    image = torch.as_tensor(image)
    mask = torch.as_tensor(mask)
    gt = torch.as_tensor(gt)

    with torch.no_grad():
        output = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask*image + (1 - mask)*output

    #grid = make_grid(
    #    torch.cat((unnormalize(image), unnormalize(mask), unnormalize(output),
    #               unnormalize(output_comp), unnormalize(gt)), dim=0))

    n = image.shape

    f = h5py.File(filename + '.hdf5', 'w')
    dset1 = f.create_dataset('image', shape=image.shape, dtype = 'float32',data = image)
    dset2 = f.create_dataset('output', shape=output.shape, dtype = 'float32',data = output)
    dset3 = f.create_dataset('output_comp', shape=output_comp.shape, dtype = 'float32',data = output_comp)
    dset4 = f.create_dataset('mask', shape=mask.shape, dtype='float32', data=mask)
    dset5 = f.create_dataset('gt', shape=gt.shape, dtype = 'float32',data = gt)
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
        image_part, mask_part, gt_part = zip(
            *[dataset[i + split * (dataset.__len__() // partitions)] for i in
              range(dataset.__len__() // partitions)])
        image_part = torch.stack(image_part)
        mask_part = torch.stack(mask_part)
        gt_part = torch.stack(gt_part)
        # get results from trained network
        with torch.no_grad():
            output_part = model(image_part.to(cfg.device), mask_part.to(cfg.device))
                                
        if cfg.lstm_steps == 0:

            image_part = image_part[:, :, :, :].to(torch.device('cpu'))
            mask_part = mask_part[:, :, :, :].to(torch.device('cpu'))
            gt_part = gt_part[:, :, :, :].to(torch.device('cpu'))
            output_part = output_part[:, :, :, :].to(torch.device('cpu'))

        else:

            image_part = image_part[:, cfg.lstm_steps - 1, :, :, :].to(torch.device('cpu'))
            mask_part = mask_part[:, cfg.lstm_steps - 1, :, :, :].to(torch.device('cpu'))
            gt_part = gt_part[:, cfg.lstm_steps - 1, :, :, :].to(torch.device('cpu'))
            output_part = output_part[:, cfg.lstm_steps - 1, :, :, :].to(torch.device('cpu'))            


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
    for var, name in zip(cvar, cname):
        h5.create_dataset(name=name, shape=var.shape, dtype=float, data=var.to(torch.device('cpu')))
        #for dim in range(0, 3):
        #    h5[cfg.data_type].dims[dim].label = dname[dim]
    h5.close()

    return ma.masked_array(gt, mask)[:, :, :, :], ma.masked_array(output_comp, mask)[:, :, :, :]



def heat_content_timeseries(depth_steps, iteration, name):

    rho = 1025  #density of seawater
    shc = 3850  #specific heat capacity of seawater

    f = h5py.File(f'{cfg.val_dir}{cfg.save_part}/validation_{iteration}_{name}.hdf5', 'r')
    output = np.array(f.get('output'))
    gt = np.array(f.get('gt'))
    image = np.array(f.get('image'))
    mask = np.array(f.get('mask'))

    continent_mask = np.where(gt[0, 0, :, :]==0, np.NaN, 1)
    mask = np.where(mask==0, np.NaN, 1)

    #take spatial mean of network output and ground truth
    masked_output = np.nanmean(np.nanmean(output * mask, axis=2), axis=2)
    masked_gt = np.nanmean(np.nanmean(image * mask, axis=2), axis=2)
    output = np.nanmean(np.nanmean(output * continent_mask, axis=2), axis=2)
    gt = np.nanmean(np.nanmean(gt * continent_mask, axis=2), axis=2)


    n = output.shape
    hc_net = np.zeros(n[0])
    hc_gt = np.zeros(n[0])
    hc_net_masked = np.zeros(n[0])
    hc_gt_masked = np.zeros(n[0])

    for i in range(n[0]):
        hc_net[i] = np.sum([(depth_steps[k] - depth_steps[k-1])*output[i, k]*rho*shc for k in range(1, n[1])]) + depth_steps[0] * output[i, 0] * rho * shc
        hc_gt[i] = np.sum([(depth_steps[k] - depth_steps[k-1])*gt[i, k]*rho*shc for k in range(1, n[1])]) + depth_steps[0] * gt[i, 0] * rho * shc
        hc_net_masked[i] = np.sum([(depth_steps[k] - depth_steps[k-1])*masked_output[i, k]*rho*shc for k in range(1, n[1])]) + depth_steps[0] * masked_output[i, 0] * rho * shc
        hc_gt_masked[i] = np.sum([(depth_steps[k] - depth_steps[k-1])*masked_gt[i, k]*rho*shc for k in range(1, n[1])]) + depth_steps[0] * masked_gt[i, 0] * rho * shc


    f_final = h5py.File(f'{cfg.val_dir}{cfg.save_part}/timeseries_{iteration}_{name}.hdf5', 'w')
    f_final.create_dataset(name='net_ts', shape=hc_net.shape, dtype=float, data=hc_net)
    f_final.create_dataset(name='gt_ts', shape=hc_gt.shape, dtype=float, data=hc_gt)
    f_final.create_dataset(name='net_ts_masked', shape=hc_net_masked.shape, dtype=float, data=hc_net_masked)
    f_final.create_dataset(name='gt_ts_masked', shape=hc_gt_masked.shape, dtype=float, data=hc_gt_masked)
    f.close()

#simple function to plot correlations between two variables
def correlation(var_1, var_2):
    
    SET = namedtuple("SET", "nsim method alpha")
    corr, significance = corr_2d_ttest(var_1, var_2, options = SET(nsim=1000, method='ttest', alpha=0.01), nd=3)
    sig = np.where(significance==True)

    return corr, significance

#function to calculate running standard deviation or mean
def running_mean_std(var, mode, del_t):

    n = var.shape
    var_out = np.zeros(n[0] - (del_t - 1))

    if mode=='mean':
        for k in range(len(var_out)):
            var_out[k] = np.nanmean(var[k:k + del_t], axis=0)
    elif mode=='std':
        for k in range(len(var_out)):
            var_out[k] = np.nanstd(var[k:k + del_t], axis=0)

    return np.array(var_out)

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
    plt.savefig(f'{cfg.save_dir}pdfs/dataset_diff{name}.pdf')
    plt.show()



def create_snapshot_image(model, dataset, filename):
    image, mask, gt = zip(*[dataset[int(i)] for i in cfg.eval_timesteps])

    image = torch.stack(image).to(cfg.device)
    mask = torch.stack(mask).to(cfg.device)
    gt = torch.stack(gt).to(cfg.device)

    with torch.no_grad():
        output = model(image, mask)

    if cfg.lstm_steps == 0:
        image = image[:, :, :, :].to(torch.device('cpu'))
        gt = gt[:,  :, :, :].to(torch.device('cpu'))
        mask = mask[:,  :, :, :].to(torch.device('cpu'))
        output = output[:,  :, :, :].to(torch.device('cpu'))

    else:
        # select last element of lstm sequence as evaluation element
        image = image[:, cfg.lstm_steps - 1, :, :, :].to(torch.device('cpu'))
        gt = gt[:, cfg.lstm_steps - 1, :, :, :].to(torch.device('cpu'))
        mask = mask[:, cfg.lstm_steps - 1, :, :, :].to(torch.device('cpu'))
        output = output[:, cfg.lstm_steps - 1, :, :, :].to(torch.device('cpu'))

    output_comp = mask * image + (1 - mask) * output

    # set mask
    mask = 1 - mask
    image = ma.masked_array(image, mask)

    mask = ma.masked_array(mask, mask)

    for c in range(output.shape[1]):
        if cfg.attribute_anomaly == 'anomalies':
            vmin, vmax = (-5, 5)
        else:
            vmin, vmax = (-10, 35)
        data_list = [image[:, c, :, :], mask[:, c, :, :], output[:, c, :, :], output_comp[:, c, :, :], gt[:, c, :, :]]

        # plot and save data
        fig, axes = plt.subplots(nrows=len(data_list), ncols=image.shape[0], figsize=(20, 20))
        fig.patch.set_facecolor('black')
        for i in range(len(data_list)):
            for j in range(image.shape[0]):
                axes[i, j].axis("off")
                axes[i, j].imshow(np.squeeze(data_list[i][j]), vmin=vmin, vmax=vmax)
        plt.subplots_adjust(wspace=0.012, hspace=0.012)
        plt.savefig(f'{filename}_{str(c)}.jpg', bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close('all')




#cfg.set_train_args()

#prepo = preprocessing(cfg.im_dir, cfg.im_name, cfg.eval_im_year, cfg.image_size, 'image', cfg.in_channels, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
#prepo.save_data()
#depths = prepo.depths()

#heat_content_timeseries_general(depths, cfg.eval_im_year)


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
from preprocessing import preprocessing
from dataloader import MaskDataset
import os


sys.path.append('./')

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

def infill(model, dataset, partitions, iter):
    if not os.path.exists(cfg.val_dir):
        os.makedirs('{:s}'.format(cfg.val_dir))
    image = []
    mask = []
    gt = []
    output = []

    if partitions > dataset.__len__():
        partitions = dataset.__len__()
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

    cvar = [image, mask, output, output_comp, gt]
    cname = ['image', 'mask', 'output', 'output_comp', 'gt']
    dname = ['time', 'lat', 'lon']
    
    h5 = h5py.File(cfg.val_dir + iter + '.hdf5', 'w')
    for x in range(0, 5):
        h5.create_dataset(name=cname[x], shape=cvar[x].shape, data=cvar[x].to(torch.device('cpu')))
        #for dim in range(0, 3):
        #    h5[cfg.data_types[0]].dims[dim].label = dname[dim]
        h5.close()

    return ma.masked_array(gt, mask)[:, :, :, :], ma.masked_array(output_comp, mask)[:, :, :, :]



def heat_content_timeseries(depth_steps, iter, plotting=False):

    rho = 1025  #density of seawater
    shc = 3850  #specific heat capacity of seawater

    f = h5py.File(cfg.val_dir + iter + '.hdf5', 'r')
    output = f.get('output_comp')
    gt = f.get('image')

    #take spatial mean of network output and ground truth
    output = np.mean(np.mean(output, axis=2), axis=2)
    gt = np.mean(np.mean(gt, axis=2), axis=2)
    n = output.shape
    hc_network = np.zeros(n[0])
    hc_assi = np.zeros(n[0])

    for i in range(n[0]):
        hc_network[i] = np.sum([(depth_steps[k] - depth_steps[k-1])*output[i, k]*rho*shc for k in range(1, n[1])]) + depth_steps[0] * output[i, 0] * rho * shc
        hc_assi[i] = np.sum([(depth_steps[k] - depth_steps[k-1])*gt[i, k]*rho*shc for k in range(1, n[1])]) + depth_steps[0] * gt[i, 0] * rho * shc


    if plotting == True:

        plt.plot(range(n[0]), hc_network, label='Network Reconstructed Heat Content')
        plt.plot(range(n[0]), hc_assi, label='Assimilation Heat Content')
        plt.grid()
        plt.legend()
        plt.xlabel('Months since January 1958')
        plt.ylabel('Heat Content [J/m²]')
        #plt.savefig('{:s}/images/{:s}/heat_content_timeseries_{}.pdf'.format(cfg.save_dir, cfg.save_part, len(depth_steps)))
        plt.show()
                



class HeatContent():

    def __init__(self, depth_steps, iter):
        self.iter = iter
        self.im_dir = cfg.im_dir
        self.im_year = cfg.eval_im_year
        self.mask_dir = cfg.mask_dir
        self.mask_year = cfg.mask_year
        self.save_path = cfg.snapshot_dir + cfg.save_part
        self.im_size = cfg.image_size
        self.mode = preprocessing
        self.attributes = [cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo]
        self.lon1 = int(cfg.lon1)
        self.lon2 = int(cfg.lon2)
        self.lat1 = int(cfg.lat1)
        self.lat2 = int(cfg.lat2)
        self.im_name = 'Image_' + str(cfg.image_name) + cfg.attribute_depth + cfg.attribute_anomaly + cfg.attribute_argo
        self.mask_name = 'Maske_' + str(cfg.mask_year) + cfg.attribute_depth + cfg.attribute_anomaly + cfg.attribute_argo
        self.shc_sw = 3850
        self.depth_steps = depth_steps

    def seawater_density(z):
        return 1025

    def creat_hc_timeseries(self, model, partitions):

        depth = True
        if self.depth != 1:
            depth = True
        
        dataset = MaskDataset(depth, self.depth, self.mask_year, self.im_year)

        gt, output_comp = infill(model, dataset, partitions)
        network = np.mean(np.mean(output_comp, axis=2), axis=2)
        assi = np.mean(np.mean(gt, axis=2), axis=2)
        n = output_comp.shape
        hc_network = np.zeros(n[0])
        hc_assi = np.zeros(n[0])

        for i in range(n[0]):
            hc_network[i] = np.sum([(self.depth_steps[k] - self.depth_steps[k-1])*network[i, k]*self.seawater_density(self.depth_steps(k))*self.shc_sw for k in range(1, n[1])])
            hc_assi[i] = np.sum([(self.depth_steps[k] - self.depth_steps[k-1])*assi[i, k]*self.seawater_density(self.depth_steps(k))*self.shc_sw for k in range(1, n[1])])

        plt.plot(range(n[0]), hc_network, label='Network Reconstructed Heat Content')
        plt.plot(range(n[0]), hc_assi, label='Assimilation Heat Content')
        plt.grid()
        plt.legend()
        plt.xlabel('Months since January 1958')
        plt.ylabel('Heat Content [J/gridcell]')
        plt.savefig('{:s}/images/{:s}/heat_content_timeseries_{:s}.pdf'.format(cfg.snapshot_dir, cfg.save_part, self.iter))
        plt.show()
            


#cfg.set_train_args()
#dataset = preprocessing(cfg.im_dir, cfg.im_name, cfg.im_year, cfg.image_size, 'image', 3, cfg.attribute_depth, cfg.attribute_anomaly, cfg.attribute_argo, cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2)
#depths = dataset.depths()
#<heat_content_timeseries('../Asi_maskiert/results/images/depth/test_550000.hdf5', depths, plotting=True)


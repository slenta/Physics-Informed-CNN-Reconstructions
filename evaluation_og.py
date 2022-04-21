from numpy.core.fromnumeric import shape
import torch
from torch.utils import data
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
from image import unnormalize
import h5py



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

    print(image.shape, mask.shape, gt.shape, output.shape)

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))

    n = image.shape

    f = h5py.File(filename + '.hdf5', 'w')
    dset1 = f.create_dataset('image', (n[0], n[1], n[2], n[3]), dtype = 'float32',data = gt)
    dset2 = f.create_dataset('output', (n[0], n[1], n[2], n[3]), dtype = 'float32',data = output)
    dset3 = f.create_dataset('output_comp', (n[0], n[1], n[2], n[3]), dtype = 'float32',data = output_comp)
    dset4 = f.create_dataset('mask', shape=(n[0], n[1], n[2], n[3]), dtype='float32', data=mask) 
    f.close()
    
    save_image(grid, filename + '.jpg')

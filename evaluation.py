import torch
from torch.utils import data
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
from image import unnormalize
import h5py



def evaluate(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(1)])
    #print(np.shape(np.array(mask)), np.shape(np.array(image)), np.shape(np.array(gt)))
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    image = torch.as_tensor(image)
    mask = torch.as_tensor(mask)
    gt = torch.as_tensor(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
output = output.to(torch.device('cpu')
    print(output.shape, np.shape(np.array(mask)), np.shape(np.array(image)), np.shape(np.array(gt)))
    output_comp = mask*image + (1 - mask)*output
    #print(output_comp.shape)
    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))

    n = image.shape
    print(n)
    f = h5py.File(filename + '.hdf5', 'w')
    dset1 = f.create_dataset('image', (n[1], n[2], n[3]), dtype = 'float32',data = image)
    dset2 = f.create_dataset('output', (n[1], n[2], n[3]), dtype = 'float32',data = output)
    dset3 = f.create_dataset('output_comp', (n[1], n[2], n[3]), dtype = 'float32',data = output_comp)
    dset4 = f.create_dataset('mask', shape=(n[1], n[2], n[3]), dtype='float32', data=mask)
    f.close()
    
    save_image(grid, filename + '.jpg')

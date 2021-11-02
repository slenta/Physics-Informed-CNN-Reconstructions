import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
from image import unnormalize



def evaluate(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    #print(np.shape(np.array(mask)), np.shape(np.array(image)), np.shape(np.array(gt)))
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    image = torch.as_tensor(image)
    mask = torch.as_tensor(mask)
    gt = torch.as_tensor(gt)
    with torch.no_grad():
        output, _ = model(image, mask)
    print(type(output), np.shape(np.array(mask)), np.shape(np.array(image)), np.shape(np.array(gt)))
    output_comp = torch.matmul(mask, image) + torch.matmul((1 - mask), output)
    print(output_comp.shape)
    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    
    print(type(grid), grid.shape)
    save_image(grid, filename)

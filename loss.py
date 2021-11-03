import torch
import torch.nn as nn


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        # get mid indexed element
        mid_index = torch.tensor([(output.shape[1] // 2)],dtype=torch.long).to('cuda')
        output = torch.index_select(output, dim=1, index=mid_index)
        output_comp = torch.index_select(output_comp, dim=1, index=torch.tensor([mid_index],dtype=torch.long))
        gt = torch.index_select(gt, dim=1, index=torch.tensor([mid_index],dtype=torch.long))

        feat_output = self.extractor(torch.cat([torch.unsqueeze(output, 1)] * 3, 1))
        feat_output_comp = self.extractor(torch.cat([torch.unsqueeze(output_comp, 1)] * 3, 1))
        feat_gt = self.extractor(torch.cat([torch.unsqueeze(gt, 1)] * 3, 1))

        loss_dict['prc'] = 0.0
        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)

        return loss_dict

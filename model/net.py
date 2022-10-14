import torch
import torch.nn as nn

from .conv_configs import init_enc_conv_configs, init_dec_conv_configs, \
    init_enc_conv_configs_orig, init_dec_conv_configs_orig
from .encoder_decoder import EncoderBlock, DecoderBlock
import config as cfg


def progstat(index, numel):
    if cfg.progress_fwd is not None:
        cfg.progress_fwd('Infilling...', int(100 * (index + 1) / numel))


class PConvLSTM(nn.Module):
    def __init__(self, img_size=128, enc_dec_layers=4, pool_layers=4, in_channels=1, out_channels=1):
        super().__init__()

        self.freeze_enc_bn = False
        self.net_depth = enc_dec_layers + pool_layers

        # initialize channel inputs and outputs and image size for encoder and decoder
        if cfg.n_filters is None:
            enc_conv_configs = init_enc_conv_configs(img_size, enc_dec_layers,
                                                     pool_layers, in_channels)
            dec_conv_configs = init_dec_conv_configs(img_size, enc_dec_layers,
                                                     pool_layers, in_channels,
                                                     out_channels)
        else:
            enc_conv_configs = init_enc_conv_configs_orig(img_size, enc_dec_layers,
                                                          out_channels, cfg.n_filters)
            dec_conv_configs = init_dec_conv_configs_orig(img_size, enc_dec_layers,
                                                          out_channels, cfg.n_filters)

        # define encoding layers
        encoding_layers = []
        for i in range(self.net_depth):
            encoding_layers.append(EncoderBlock(
                conv_config=enc_conv_configs[i],
                kernel=enc_conv_configs[i]['kernel'], stride=(2, 2), activation=nn.ReLU()))
        self.encoder = nn.ModuleList(encoding_layers)

        # define decoding layers
        decoding_layers = []
        for i in range(self.net_depth):
            if i == self.net_depth - 1:
                activation = None
                bias = True
            else:
                activation = nn.LeakyReLU()
                bias = False
            decoding_layers.append(DecoderBlock(
                conv_config=dec_conv_configs[i],
                kernel=dec_conv_configs[i]['kernel'], stride=(1, 1), activation=activation, bias=bias))
        self.decoder = nn.ModuleList(decoding_layers)

    def forward(self, input, input_mask):
        # create lists for skip connections
        h = input
        h_mask = input_mask
        hs = [h]
        hs_mask = [h_mask]
        recurrent_states = []

        # forward pass encoding layers
        for i in range(self.net_depth):
            h, h_mask, recurrent_state = self.encoder[i](hs[i],
                                                         hs_mask[i],
                                                         None)
            # save hidden states for skip connections
            hs.append(h)
            recurrent_states.append(recurrent_state)
            hs_mask.append(h_mask)

            progstat(i, 2 * self.net_depth)

        # reverse all hidden states
        if cfg.recurrent_steps:
            for i in range(self.net_depth):
                hs[i] = torch.flip(hs[i], (1,))
                hs_mask[i] = torch.flip(hs_mask[i], (1,))

        h, h_mask = hs[self.net_depth], hs_mask[self.net_depth]

        # forward pass decoding layers
        for i in range(self.net_depth):
            if cfg.recurrent_steps:
                h, h_mask, recurrent_state = self.decoder[i](h, hs[self.net_depth - i - 1],
                                                             h_mask, hs_mask[self.net_depth - i - 1],
                                                             recurrent_states[self.net_depth - 1 - i])
            else:
                h, h_mask, recurrent_state = self.decoder[i](h, hs[self.net_depth - i - 1],
                                                             h_mask, hs_mask[self.net_depth - i - 1],
                                                             None)
            progstat(i + self.net_depth, 2 * self.net_depth)

        # return last element of output from last decoding layer
        return h

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc_bn:
            for i in range(self.net_depth):
                if hasattr(self.encoder[i].partial_conv, "bn"):
                    if isinstance(self.encoder[i].partial_conv.bn, nn.BatchNorm2d):
                        self.encoder[i].eval()

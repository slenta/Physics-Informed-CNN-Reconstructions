import torch
import torch.nn as nn
import sys

sys.path.append('./')
from model.attention_module import AttentionEncoderBlock
from model.encoder_decoder import EncoderBlock, DecoderBlock
from model.conv_configs import init_enc_conv_configs, init_dec_conv_configs
import config as cfg


class PConvLSTM(nn.Module):
    def __init__(self, radar_img_size=128, radar_enc_dec_layers=4, radar_pool_layers=4, radar_in_channels=1,
                 radar_out_channels=1,
                 rea_img_size=None, rea_enc_layers=None, rea_pool_layers=None, rea_in_channels=0,
                 lstm=True):
        super().__init__()

        self.freeze_enc_bn = False
        self.net_depth = radar_enc_dec_layers + radar_pool_layers
        self.lstm = lstm

        # initialize channel inputs and outputs and image size for encoder and decoder
        enc_conv_configs = init_enc_conv_configs(radar_img_size, radar_enc_dec_layers,
                                                 radar_pool_layers, radar_in_channels)
        dec_conv_configs = init_dec_conv_configs(radar_img_size, radar_enc_dec_layers,
                                                 radar_pool_layers, radar_in_channels,
                                                 radar_out_channels)

        if rea_img_size:
            self.channel_fusion_depth = rea_enc_layers + rea_pool_layers
            enc_conv_configs[self.net_depth - self.channel_fusion_depth]['in_channels'] += rea_in_channels
            dec_conv_configs[self.channel_fusion_depth - 1]['skip_channels'] += cfg.skip_layers*rea_in_channels

        # define encoding layers
        encoding_layers = []
        for i in range(0, self.net_depth):
            if i == 0:
                kernel = (7, 7)
            elif i < radar_enc_dec_layers - 1:
                kernel = (5, 5)
            else:
                kernel = (5, 5)
            encoding_layers.append(EncoderBlock(
                conv_config=enc_conv_configs[i],
                kernel=kernel, stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
        self.encoder = nn.ModuleList(encoding_layers)

        # define decoding layers
        decoding_layers = []
        for i in range(self.net_depth):
            if i == self.net_depth - 1:
                activation = None
                bn = False
                bias = True
            else:
                activation = nn.LeakyReLU()
                bn = True,
                bias = False
            decoding_layers.append(DecoderBlock(
                conv_config=dec_conv_configs[i],
                kernel=(3, 3), stride=(1, 1), activation=activation, lstm=lstm, bn=bn, bias=bias))
        self.decoder = nn.ModuleList(decoding_layers)

    def forward(self, input, input_mask):
        # create lists for skip connections
        h = input
        h_mask = input_mask
        hs = [h]
        hs_mask = [h_mask]
        lstm_states = []

        # forward pass encoding layers
        for i in range(self.net_depth):
            h, h_mask, lstm_state = self.encoder[i](hs[i],
                                                    hs_mask[i],
                                                    None)

            # save hidden states for skip connections
            hs.append(h)
            lstm_states.append(lstm_state)
            hs_mask.append(h_mask)

        # reverse all hidden states
        for i in range(self.net_depth):
            hs[i] = torch.flip(hs[i], (1,))
            hs_mask[i] = torch.flip(hs_mask[i], (1,))

        h, h_mask = hs[self.net_depth], hs_mask[self.net_depth]

        # forward pass decoding layers
        for i in range(self.net_depth):
            lstm_state_h, lstm_state_c = None, None
            if self.lstm:
                lstm_state_h, lstm_state_c = lstm_states[self.net_depth - 1 - i]
            h, h_mask, lstm_state = self.decoder[i](h, hs[self.net_depth - i - 1],
                                                    h_mask, hs_mask[self.net_depth - i - 1],
                                                    (lstm_state_h, lstm_state_c))
            

        # return last element of output from last decoding layer

        return h

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc_bn:
            for i in range(self.net_depth):
                if isinstance(self.encoder[i].partial_conv.bn, nn.BatchNorm2d):
                    self.encoder[i].eval()

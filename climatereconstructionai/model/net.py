import torch
import torch.nn as nn

from .attention_module import AttentionEncoderBlock
from .conv_configs import init_enc_conv_configs, init_dec_conv_configs, \
    init_enc_conv_configs_orig, init_dec_conv_configs_orig
from .encoder_decoder import EncoderBlock, DecoderBlock
from .. import config as cfg


def progstat(index, numel):
    if cfg.progress_fwd:
        f = open(cfg.log_dir + "/progfwd.info", "w")
        print(int(100 * (index + 1) / numel), file=f)
        f.close()


class PConvLSTM(nn.Module):
    def __init__(self, radar_img_size=512, radar_enc_dec_layers=4, radar_pool_layers=4, radar_in_channels=1,
                 radar_out_channels=1,
                 rea_img_size=None, rea_enc_layers=None, rea_pool_layers=None, rea_in_channels=0,
                 recurrent=True):
        super().__init__()

        self.freeze_enc_bn = False
        self.net_depth = radar_enc_dec_layers + radar_pool_layers
        self.recurrent = recurrent

        # initialize channel inputs and outputs and image size for encoder and decoder
        if cfg.n_filters is None:
            enc_conv_configs = init_enc_conv_configs(radar_img_size, radar_enc_dec_layers,
                                                     radar_pool_layers, radar_in_channels)
            dec_conv_configs = init_dec_conv_configs(radar_img_size, radar_enc_dec_layers,
                                                     radar_pool_layers, radar_in_channels,
                                                     radar_out_channels)
        else:
            enc_conv_configs = init_enc_conv_configs_orig(radar_img_size, radar_enc_dec_layers,
                                                          radar_out_channels, cfg.n_filters)
            dec_conv_configs = init_dec_conv_configs_orig(radar_img_size, radar_enc_dec_layers,
                                                          radar_out_channels, cfg.n_filters)

        if cfg.attention:
            self.attention_depth = rea_enc_layers + rea_pool_layers
            attention_enc_conv_configs = init_enc_conv_configs(rea_img_size, rea_enc_layers,
                                                               rea_pool_layers, rea_in_channels)
            attention_layers = []
            for i in range(self.attention_depth):
                if i < rea_enc_layers:
                    kernel = (5, 5)
                else:
                    kernel = (3, 3)
                attention_layers.append(AttentionEncoderBlock(
                    conv_config=attention_enc_conv_configs[i],
                    kernel=kernel, stride=(2, 2), activation=nn.ReLU()))

                # adjust skip channels for decoder
                if i != self.attention_depth - 1:
                    dec_conv_configs[i]['out_channels'] += \
                        attention_enc_conv_configs[self.attention_depth - i - 1]['in_channels']
                dec_conv_configs[i]['skip_channels'] += \
                    cfg.skip_layers * attention_enc_conv_configs[self.attention_depth - i - 1]['in_channels']
                dec_conv_configs[i]['in_channels'] += \
                    attention_enc_conv_configs[self.attention_depth - i - 1]['out_channels']

            self.attention_module = nn.ModuleList(attention_layers)

        elif rea_img_size:
            self.channel_fusion_depth = rea_enc_layers + rea_pool_layers
            enc_conv_configs[self.net_depth - self.channel_fusion_depth]['in_channels'] += rea_in_channels
            dec_conv_configs[self.channel_fusion_depth - 1]['skip_channels'] += cfg.skip_layers * rea_in_channels

        # define encoding layers
        encoding_layers = []
        for i in range(0, self.net_depth):
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

    def forward(self, input, input_mask, attention_input, attention_input_mask):
        # create lists for skip connections
        h = input
        h_mask = input_mask
        hs = [h]
        hs_mask = [h_mask]
        recurrent_states = []

        h_attention = attention_input
        h_attention_mask = attention_input_mask
        attentions = []
        attentions_mask = []
        attentions_recurrent_states = []

        # forward pass encoding layers
        for i in range(self.net_depth):

            if h_attention.size()[1] != 0 and h.shape[3] == h_attention.shape[3]:
                if not cfg.attention:
                    hs[i] = torch.cat([hs[i], h_attention], dim=2)
                    hs_mask[i] = torch.cat([hs_mask[i], h_attention_mask], dim=2)

            h, h_mask, recurrent_state = self.encoder[i](hs[i],
                                                         hs_mask[i],
                                                         None)

            # execute attention module if configured
            if cfg.attention and i >= (self.net_depth - self.attention_depth):
                attention_index = i - (self.net_depth - self.attention_depth)
                h_attention, h_attention_mask, attention_recurrent_state, attention = \
                    self.attention_module[attention_index](h_attention,
                                                           h_attention_mask,
                                                           None,
                                                           h)
                attentions.append(attention)
                attentions_mask.append(h_attention_mask)
                attentions_recurrent_states.append(attention_recurrent_state)

            # save hidden states for skip connections
            hs.append(h)
            recurrent_states.append(recurrent_state)
            hs_mask.append(h_mask)

            progstat(i, 2 * self.net_depth)

        # concat attentions
        if cfg.attention:
            hs[self.net_depth - self.attention_depth] = torch.cat(
                [hs[self.net_depth - self.attention_depth], attention_input], dim=2)
            hs_mask[self.net_depth - self.attention_depth] = torch.cat(
                [hs_mask[self.net_depth - self.attention_depth], attention_input_mask], dim=2)
            for i in range(self.attention_depth):
                hs[i + (self.net_depth - self.attention_depth) + 1] = torch.cat(
                    [hs[i + (self.net_depth - self.attention_depth) + 1], attentions[i]], dim=2)
                hs_mask[i + (self.net_depth - self.attention_depth) + 1] = torch.cat(
                    [hs_mask[i + (self.net_depth - self.attention_depth) + 1], attentions_mask[i]], dim=2)

                if cfg.lstm_steps:
                    lstm_state_h, lstm_state_c = recurrent_states[i + (self.net_depth - self.attention_depth)]
                    attention_lstm_state_h, attention_lstm_state_c = attentions_recurrent_states[i]
                    lstm_state_h = torch.cat([lstm_state_h, attention_lstm_state_h], dim=1)
                    lstm_state_c = torch.cat([lstm_state_c, attention_lstm_state_c], dim=1)
                    recurrent_states[i + (self.net_depth - self.attention_depth)] = (lstm_state_h, lstm_state_c)

        # reverse all hidden states
        if self.recurrent:
            for i in range(self.net_depth):
                hs[i] = torch.flip(hs[i], (1,))
                hs_mask[i] = torch.flip(hs_mask[i], (1,))

        h, h_mask = hs[self.net_depth], hs_mask[self.net_depth]

        # forward pass decoding layers
        for i in range(self.net_depth):
            if self.recurrent:
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

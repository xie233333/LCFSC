import torch.nn as nn
import numpy as np
import os
import torch
import time


class Channel(nn.Module):
    def __init__(self, args, config):
        super(Channel, self).__init__()
        self.config = config
        if config.logger:
            config.logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                args.channel_type, args.multiple_snr))


    def forward(self, input, chan_param, h):       
        channel_in = input
        chan_param = 10 ** (chan_param / 10)

        xpower = torch.sum(channel_in.real ** 2 + channel_in.imag ** 2) / channel_in.numel()
        npower = xpower / chan_param / 2
        npower = npower.to(self.config.device)
        noise_real = torch.randn(channel_in.shape).to(self.config.device) * torch.sqrt(npower)
        noise_imag = torch.randn(channel_in.shape).to(self.config.device) * torch.sqrt(npower)
        noise = noise_real + 1j * noise_imag
        channel_output = torch.matmul(h, channel_in) + noise.to(self.config.device)
        return channel_output
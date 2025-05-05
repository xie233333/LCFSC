#!/usr/bin/env python3
import torch.nn as nn
import numpy as np
import torch
import scipy.io as scio


def conv3x3(in_chn, out_chn):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1)
    return layer


def conv_down(in_chn, out_chn):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1)
    return layer

class UNetD(nn.Module):

    def __init__(self, in_chn, wf=16, depth=3, relu_slope=0.2, subspace_dim=16):
        super(UNetD, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        prev_channels = self.get_input_chn(in_chn)
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, downsample, relu_slope))
            prev_channels = (2**i) * wf

        # self.ema = EMAU(prev_channels, prev_channels//8)
        self.up_path = nn.ModuleList()
        subnet_repeat_num = 1
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope, subnet_repeat_num, subspace_dim))
            prev_channels = (2**i)*wf
            subnet_repeat_num += 1

        self.last = conv3x3(prev_channels, in_chn)
                    
        self.pred = nn.Sequential(
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, 128), # 2*6
        nn.Unflatten(dim=1, unflattened_size=(2, 8, 8)),
        )                                


    def forward(self, x1):
        blocks = []
        for i, down in enumerate(self.down_path):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                blocks.append(x1_up)
            else:
                x1 = down(x1)
        # x1 = self.ema(x1)
        for i, up in enumerate(self.up_path):
            x1 = up(x1, blocks[-i-1])

        pred = self.last(x1)
        pred = self.pred(pred)
        return pred

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print("weight")
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    print("bias")
                    nn.init.zeros_(m.bias)


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, downsample, relu_slope):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            nn.LeakyReLU(relu_slope))

        self.downsample = downsample
        if downsample:
            self.downsample = conv_down(out_size, out_size)

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope, subnet_repeat_num, subspace_dim=16):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)
        self.num_subspace = subspace_dim

        self.subnet = Subspace(in_size, self.num_subspace)
        self.skip_m = skip_blocks(out_size, out_size, subnet_repeat_num)

    def forward(self, x, bridge):
        up = self.up(x)
        bridge = self.skip_m(bridge)
        out = torch.cat([up, bridge], 1)
        if self.subnet:
            b_, c_, h_, w_ = bridge.shape
            sub = self.subnet(out)
            V_t = sub.reshape(b_, self.num_subspace, h_*w_)
            V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdims=True))
            V = V_t.permute(0, 2, 1)
            mat = torch.matmul(V_t, V)
            mat_inv = torch.linalg.inv(mat)
            project_mat = torch.matmul(mat_inv, V_t)
            bridge_ = bridge.reshape(b_, c_, h_*w_)
            project_feature = torch.matmul(project_mat, bridge_.permute(0, 2, 1))
            bridge = torch.matmul(V, project_feature).permute(0, 2, 1).reshape(b_, c_, h_, w_)
            out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(UNetConvBlock(in_size, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1)

    def forward(self, x):
        sc = self.shortcut(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.re_num = repeat_num
        mid_c = 64
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))
        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1)

    def forward(self, x):
        sc = self.shortcut(x)
        for m in self.blocks:
            x = m(x)
        return x + sc

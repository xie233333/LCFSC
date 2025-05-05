import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class SMNet(nn.Module):
    def __init__(self, input_resolution, dim):
        super(SMNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Unflatten(dim=0, unflattened_size=(1, input_resolution[0], input_resolution[1])),
            nn.Conv2d(1, 2, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(2, 4, 3, stride=1, padding=1),
            nn.GELU()
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(128, input_resolution[0]*input_resolution[1]),
            nn.GELU(),
            nn.Linear(input_resolution[0]*input_resolution[1], 4*input_resolution[0]*input_resolution[1]),
            nn.GELU()
        )

        self.encoder2 = nn.Sequential(
            nn.Unflatten(dim=0, unflattened_size=(4, 1*input_resolution[0], 1*input_resolution[1])),
            nn.Conv2d(4, 2, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(2, 4, 3, stride=1, padding=1),
            nn.GELU()
        )

        self.simple_conv = nn.Sequential(
            nn.Conv2d(8, 4, 3, stride=1, padding=1),
            nn.GELU()               
        )

        self.fc2 = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(4*input_resolution[0]*input_resolution[1], 2*input_resolution[0]*input_resolution[1]),
            nn.GELU(),                  
            nn.Linear(2*input_resolution[0]*input_resolution[1], 64*64)         
            )


        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def ste_round(self, x):
        return torch.round(x) - x.detach() + x

    def forward(self, x, H, y, tra):
        masked_ratio = torch.tensor([0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.014, 0.015]).cuda()
        # masked_ratio = torch.tensor([0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1]).cuda()
        masked_ratio = masked_ratio.to(torch.float32)
        n = x.detach().size()
        encode1 = self.encoder1(x)
        H_R = H.real
        H_I = H.imag
        H = torch.concat((H_R, H_I), dim=0)
        h_ = self.fc1(H)
        encode2 = self.encoder2(h_)

        out = torch.concat((encode1, encode2), dim=0)
        out = self.simple_conv(out)
        out = self.fc2(out)
        
        # mr = masked_ratio[torch.argmax(y)]

        mr = torch.mul(y, masked_ratio).cuda()
        mr = torch.sum(mr)

        if tra == 0:
            print(mr)
        value = mr * 4096
        
        out = F.softmax(out, dim=-1)
        out = out.argsort() # 从小到大排序，输出序号
        
        out = out.view(64, 64)
        out = out.float()
        a = torch.zeros(out.shape).cuda()
        b = torch.ones(out.shape).cuda()
        out = torch.where(out > value, out, a)
        out = torch.where(out < value, out, b)
        return out

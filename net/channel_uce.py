import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_

class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dim, apply_batchnorm=True, add = False):
        super().__init__()

        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        ac = nn.LeakyReLU()
        self.encoder_layer = None
        if add:
            self.encoder_layer = nn.Sequential(conv)
        elif apply_batchnorm:
            bn = nn.BatchNorm2d(dim)
            self.encoder_layer = nn.Sequential(conv, bn, ac)
        else:
            self.encoder_layer = nn.Sequential(conv, ac)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        return self.encoder_layer(x)


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, dim, apply_dropout=False, add = False):
        super().__init__()
        
        dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        bn = nn.BatchNorm2d(dim)
        ac = nn.GELU()
        self.decoder_layer = None
        
        if add:
            self.decoder_layer = nn.Sequential(dconv)      
        elif apply_dropout:
            drop = nn.Dropout(0.5)
            self.decoder_layer = nn.Sequential(dconv, bn, drop, ac)
        else:
            self.decoder_layer = nn.Sequential(dconv, bn, ac)
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()    
        
            

    def forward(self, x):
        return self.decoder_layer(x)
    
    


class Channel_Estimator(nn.Module):
    def __init__(self):
        super().__init__()

        p_layer_1  = DecoderLayer(in_channels=2, out_channels=4, kernel_size=2, stride=2, padding=0, output_padding=0, dim=4) # 4*16*16

        self.p_layers = nn.ModuleList([p_layer_1])

        #encoder
        encoder_layer_1 = EncoderLayer(in_channels=4, out_channels=8, kernel_size=2, stride=2, padding=0, dim=8)   # 8*8*8
        encoder_layer_2 = EncoderLayer(in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=0, dim=16)  # 16*4*4
        encoder_layer_3 = EncoderLayer(in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=0, dim=32)  # 32*2*2

        self.encoder_layers = nn.ModuleList([encoder_layer_1, encoder_layer_2, encoder_layer_3])

        # deconder
        decoder_layer_1 = DecoderLayer(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0, output_padding=0, dim=16) # 16*4*4 
        decoder_layer_2 = DecoderLayer(in_channels=16, out_channels=8, kernel_size=2, stride=2, padding=0, output_padding=0, dim=8) # 8*8*8  
        self.decoder_layers = nn.ModuleList([decoder_layer_1, decoder_layer_2])

        self.last = nn.Sequential(
            nn.Conv2d(8, 4, 2, 2, 0),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(192, 128),
            nn.Unflatten(dim=1, unflattened_size=(2, 8, 8)),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # pass the encoder and record xs
        for p_layer in self.p_layers:
            x = p_layer(x)

        encoder_xs = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encoder_xs.append(x)
        encoder_xs = encoder_xs[:-1][::-1]    # reverse

        # pass the decoder and apply skip connection
        for i, decoder_layer in enumerate(self.decoder_layers):
            x = decoder_layer(x)
            x = torch.cat([x, encoder_xs[i]], axis=-1)     # skip connect
        
        return self.last(x)        # last
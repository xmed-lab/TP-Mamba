from .sam_utils import desequence, sequence
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.blocks.dynunet_block import UnetResBlock
import torch
import torch.nn as nn
import math

act_params = ("gelu")

class Conv3D_UP(nn.Module):
    def __init__(self, in_size, out_size, scale=(2, 2, 1), kernel_size=(1, 1, 3), padding_size=(0, 0, 1), init_stride=(1, 1, 1)):
        super(Conv3D_UP, self).__init__()
        self.convup = nn.Sequential(
                                    Convolution(3, in_size, out_size, strides=scale, kernel_size=kernel_size, act=act_params, is_transposed=True, adn_ordering='NDA'),
                                    ResidualUnit(3, out_size, out_size, strides=1, kernel_size=kernel_size, subunits=1, act=act_params, adn_ordering='NDA')
                                    )

    def forward(self, inputs):
        outputs = self.convup(inputs)
        return outputs

class SAM_Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes=1, downsample_rate=8.0):
        super(SAM_Decoder, self).__init__()
        
        #### 2x upsample features from the last four blocks in SAM_ViT
        self.up4 = Conv3D_UP(in_dim, out_dim, (2, 2, 1), 3, 1)
        self.up3 = Conv3D_UP(in_dim, out_dim, (2, 2, 1), 3, 1)
        self.up2 = Conv3D_UP(in_dim, out_dim, (2, 2, 1), 3, 1)
        self.up1 = Conv3D_UP(in_dim, out_dim, (2, 2, 1), 3, 1)
        
        #### upsample gathered features back to original resolution
        num_upsample = int(math.log2(downsample_rate) - 1)
        layers = [UnetResBlock(3, out_dim*4, out_dim, 3, stride=1, act_name=act_params, norm_name="instance")] + [Conv3D_UP(out_dim, out_dim, (2, 2, 1), 3, 1) for i in range(num_upsample)] 
   
        self.up = nn.Sequential(*layers)
    
        self.final = nn.Conv3d(out_dim, num_classes, 1)
    
    def forward(self, x_lis, batch, depth):
        x4 = sequence(x_lis[-1], batch, depth)
        x4 = self.up4(x4)
        x3 = sequence(x_lis[-2], batch, depth)
        x3 = self.up3(x3)
        x2 = sequence(x_lis[-3], batch, depth)
        x2 = self.up2(x2)
        x1 = sequence(x_lis[-4], batch, depth)
        x1 = self.up1(x1)
        
        x = torch.cat([x1,x2,x3,x4], dim=1)
        x = self.up(x)
        
        output = self.final(x)
        
        return output
        

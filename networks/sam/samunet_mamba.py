import timm
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import PatchEmbed, Mlp, DropPath, PatchDropout, LayerNorm2d, ClassifierHead, NormMlpClassifierHead,\
    Format, resample_abs_pos_embed_nhwc, RotaryEmbeddingCat, apply_rot_embed_cat, to_2tuple, use_fused_attn
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import get_conv_layer, UnetOutBlock, UnetBasicBlock, UnetResBlock
from monai.networks.blocks import UpSample, Convolution, ResidualUnit
from typing import Callable, Optional, Tuple
from mamba_ssm import Mamba
activation = nn.GELU
# from networks.networks_other import init_weights
from torch.nn import init
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, hw: Tuple[int, int], pad_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    Hp, Wp = pad_hw if pad_hw is not None else hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    x = x[:, :H, :W, :].contiguous()
    return x

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
def block_forward(x, block, adapter1, adapter2):
    B, H, W, _ = x.shape

    x = adapter1(x)

    shortcut = x
    x = block.norm1(x)
    if block.window_size > 0:
        x, pad_hw = window_partition(x, block.window_size)
    x = block.drop_path1(block.ls1(block.attn(x)))
    if block.window_size > 0:
        x = window_unpartition(x, block.window_size, (H, W), pad_hw)
    x = shortcut + x

    x = adapter2(x)

    x = x.reshape(B, H * W, -1)  # MLP is faster for N, L, C tensor
    x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
    x = x.reshape(B, H, W, -1)
    return x


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

class UnetConv3UP(nn.Module):
    def __init__(self, in_size, out_size, scale=(2,2,2), kernel_size=(1,1,3), padding_size=(0,0,1), init_stride=(1,1,1)):
        super(UnetConv3UP, self).__init__()
        self.convup = nn.Sequential(#nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       #nn.InstanceNorm3d(out_size),
                                       #act_func,
                                       #nn.Upsample(scale_factor=scale, mode='trilinear'),
                                       Convolution(3, in_size, out_size, strides=scale, kernel_size=kernel_size, act=act_params, is_transposed=True, adn_ordering='NDA'),
                                       ResidualUnit(3, out_size, out_size, strides=1, kernel_size=kernel_size, subunits=1, act=act_params, adn_ordering='NDA')
                                       )

    def forward(self, inputs):
        outputs = self.convup(inputs)
        return outputs
    
class UnetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, scale=(2,2,2), kernel_size=(1,1,3)):
        super(UnetUpBlock, self).__init__()
        self.up = UnetConv3UP(in_size, out_size, scale, kernel_size)
        self.merge = UnetResBlock(3, out_size*2, out_size, kernel_size, stride=1, act_name=act_params, norm_name="instance")
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.merge(x)
        return x

class _LoRA_qkv_timm(nn.Module):
    def __init__(
        self,
        qkv: nn.Module,
        r: int
    ):
        super().__init__()
        self.r = r
        self.qkv = qkv
        self.dim = qkv.in_features
        self.linear_a_q = nn.Linear(self.dim, r, bias=False)
        self.linear_b_q = nn.Linear(r, self.dim, bias=False)
        self.linear_a_v = nn.Linear(self.dim, r, bias=False)
        self.linear_b_v = nn.Linear(r, self.dim, bias=False)
        self.act = act_func
        self.w_identity = torch.eye(self.dim)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.linear_a_q.weight, a=math.sqrt(5))     
        nn.init.kaiming_uniform_(self.linear_a_v.weight, a=math.sqrt(5))  
        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.zeros_(self.linear_b_v.weight)
    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.act(self.linear_a_q(x)))
        new_v = self.linear_b_v(self.act(self.linear_a_v(x)))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv
class CIR(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, padding, dilation=1,groups=1):
        super(CIR, self).__init__(
            nn.Conv3d(in_channels, out_channels, kernel, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.InstanceNorm3d(out_channels),
            act_func,
        )
class adapter3(nn.Module):
    def __init__(self, mode=0):
        super().__init__()
        self.dim = 768
        r = self.dim // 4
        self.mode = mode
        self.down = nn.Linear(self.dim, r, bias=False)
        self.up = nn.Linear(r, self.dim, bias=False)
        self.mid = nn.Sequential(
                    #nn.LayerNorm(r),
                    Mamba(d_model=r, d_state=16, d_conv=4, expand=2),
                    nn.LayerNorm(r),
                    act_func
                )      
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))  
        nn.init.zeros_(self.up.weight)
    def desequence(self, x):
        self.batch, channel, h, w, self.depth = x.size()
        x = x.permute(0,4,2,3,1).reshape(-1, h, w, channel)
        return x
    def sequence(self, x, b=1):
        bd, h, w, channel = x.size()
        x = x.reshape(b, bd//b, h, w, channel)
        return x.permute(0,4,2,3,1)
    def forward(self, x, b=1):
        shortcut = x
        x = self.down(x)
        bd, h, w, channel = x.size()
        d = bd // b
        if self.mode == 0:
            x = x.reshape(b, d, h, w, channel).reshape(b, -1, channel)
            x = self.mid(x)
            x = x.reshape(b, d, h, w, channel).reshape(bd, h, w, channel)
        elif self.mode == 1:
            x = x.reshape(bd, -1, channel)
            x = self.mid(x)
            x = x.reshape(bd, h, w, channel)
        elif self.mode == 2:
            x = x.reshape(b, d, h, w, channel).transpose(1,2).reshape(b*h, -1, channel)
            x = self.mid(x)
            x = x.reshape(b, h, d, w, channel).transpose(1,2).reshape(bd, h, w, channel)
        x = self.up(x)
        x = shortcut + x
        return x
    
#act_func = nn.PReLU(num_parameters=1)
act_func = nn.GELU()
#act_params = ("leakyrelu", {"negative_slope": 0.1, "inplace": True})
act_params = ("gelu")
import time
class SAM_Mamba(nn.Module):
    def __init__(self, num_classes, mode=0, kernel=3):
        super().__init__()
        droppath = 0.0
        model = timm.create_model(
                    'samvit_base_patch16.sa1b',
                    pretrained=True,
                    num_classes=0,  # remove classifier nn.Linear
                )
        for name, param in model.named_parameters():
            #if "patch_embed" in name:
            #    print("learnable patch_embed")
            #    param.requires_grad = True
            #else:
            param.requires_grad = False
        Adapter = []
        self.begin = 8
        self.len = 12
        for t_layer_i, blk in enumerate(model.blocks):
            if t_layer_i < self.len:
                model.blocks[t_layer_i].attn.qkv = _LoRA_qkv_timm(blk.attn.qkv, 16*4)
            else:
                model.blocks[t_layer_i] = nn.Identity()
        model.neck = nn.Identity()
        print("the len {} and begin from {} and kernel_size {} samunetconv ".format(self.len, self.begin, kernel))
        print("samunetconv_{}".format(kernel))
        #self.Adapter1 = nn.Sequential(*[adapter3(kernel) for i in range(self.len)]) 
        self.Adapter2 = nn.Sequential(*[adapter3(mode) for i in range(self.len)]) 
        self.sam = model
        del model
        self.num_classes = num_classes
        out_dim = 64
        #self.patch = nn.Conv3d(1, 768, kernel_size=(8,8,5), stride=(8,8,1), padding=(0,0,2), bias=True)
        #nn.init.zeros_(self.patch.weight)
        #nn.init.zeros_(self.patch.bias)
        print("no_patch with out_dim {}".format(out_dim))
        self.up = nn.Sequential(
                   UnetResBlock(3, out_dim*4, out_dim*2, 3, stride=1, act_name=act_params, norm_name="instance"),
                   UnetConv3UP(out_dim*2, out_dim, (2,2,1), 3, 1),
                   UnetConv3UP(out_dim, out_dim, (2,2,1), 3, 1),
                   #Convolution(3, out_dim, out_dim, kernel_size=3, adn_ordering='NDA', dropout=0.1, act=act_params),
        )
        self.up4 = UnetConv3UP(768, out_dim, (2,2,1), 3, 1)
        self.up3 = UnetConv3UP(768, out_dim, (2,2,1), 3, 1)
        self.up2 = UnetConv3UP(768, out_dim, (2,2,1), 3, 1)
        self.up1 = UnetConv3UP(768, out_dim, (2,2,1), 3, 1)
        print(self.up)
        #self.up1 = UnetrUpBlock(spatial_dims=3, in_channels=768, out_channels=out_dim, kernel_size=3, upsample_kernel_size=(2,2,1), norm_name="instance", res_block=True)
        self.final = nn.Conv3d(out_dim, num_classes, 1)
    def desequence(self, x):
        self.batch, channel, h, w, self.depth = x.size()
        x = x.permute(0,4,1,2,3).reshape(-1, channel, h, w)
        return x
    def sequence(self, x):
        bd, channel, h, w = x.size()
        x = x.reshape(self.batch, self.depth, channel, h, w)
        return x.permute(0,2,3,4,1)
    def forward_encoder(self, x):
        #x_new = self.patch(x)
        #x_new = x_new.permute(0,4,2,3,1).reshape(-1, 16, 16, 768)
        x = F.interpolate(x, scale_factor=(2,2,1),mode='trilinear', align_corners=False)
        x = x.expand(-1,3,-1,-1,-1)
        x = self.desequence(x)
        x = self.sam.patch_embed(x)
        if self.sam.pos_embed is not None:
            x = x + resample_abs_pos_embed_nhwc(self.sam.pos_embed, x.shape[1:3])
        x = self.sam.pos_drop(x)
        x = self.sam.patch_drop(x)
        x = self.sam.norm_pre(x)

        res = []
        for i in range(self.len):
            #x = self.Adapter1[i](x)
            x = self.sam.blocks[i](x)
            x = self.Adapter2[i](x, self.batch)
            #x = block_forward(x, self.sam.blocks[i], self.Adapter1[i], self.Adapter2[i], self.batch)
            if i >= self.begin:
                res.append(x.permute(0, 3, 1, 2))
        return res
    def forward_decoder(self, x_lis):
        x4 = self.sequence(x_lis[-1])
        x4 = self.up4(x4)
        x3 = self.sequence(x_lis[-2])
        x3 = self.up3(x3)
        x2 = self.sequence(x_lis[-3])
        x2 = self.up2(x2)
        x1 = self.sequence(x_lis[-4])
        x1 = self.up1(x1)
        x = torch.cat([x1,x2,x3,x4], dim=1)
        x = self.up(x)
        output = self.final(x)
        return output
    def forward(self, x):
        features = self.forward_encoder(x)
        output = self.forward_decoder(features)
        return output

from .sam_utils import window_partition, window_unpartition, desequence, sequence, init_weights, weights_init_kaiming
import torch
import torch.nn as nn
import math

act_func = nn.GELU()
act_params = ("gelu")

class CIR(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, padding, dilation=1,groups=1):
        super(CIR, self).__init__(
            nn.Conv3d(in_channels, out_channels, kernel, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.InstanceNorm3d(out_channels),
            act_func,
        )

class Adapter_MSConv(nn.Module):
    def __init__(self, kernel = 3):
        super().__init__()
        self.dim = 768
        r = self.dim // 4
        ratio = kernel // 2
        dilation = [1, 2, 4, 8]
        
        self.down = CIR(self.dim, r, (1, 1, kernel), (0, 0, ratio))
        
        self.b1 = CIR(r, r//4, (1, 1, kernel), (0, 0, dilation[0]*ratio), dilation[0])
        self.b2 = CIR(r, r//4, (1, 1, kernel), (0, 0, dilation[1]*ratio), dilation[1])
        self.b3 = CIR(r, r//4, (1, 1, kernel), (0, 0, dilation[2]*ratio), dilation[2])
        self.b4 = CIR(r, r//4, (1, 1, kernel), (0, 0, dilation[3]*ratio), dilation[3])
        
        self.up = nn.Conv3d(r, self.dim, (1, 1, kernel), padding=(0, 0, ratio), bias=False)

        self.down.apply(weights_init_kaiming)
        self.b1.apply(weights_init_kaiming)
        self.b2.apply(weights_init_kaiming)
        self.b3.apply(weights_init_kaiming)
        self.b4.apply(weights_init_kaiming) 
        nn.init.zeros_(self.up.weight)

    def forward(self, x, b=1, d=96):
        shortcut = x
        x = sequence(x, b, d)
        x = self.down(x)
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        x = self.up(x)
        x = desequence(x)
        x = shortcut + x
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
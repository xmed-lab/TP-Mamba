from .sam_decoder import SAM_Decoder
from .sam_utils import desequence
from .adapters import _LoRA_qkv_timm as LoRA
from .adapters import Adapter_MSConv as MSConv
import torch
import torch.nn as nn
import timm
from timm.layers import resample_abs_pos_embed_nhwc

class SAM_MS(nn.Module):
    def __init__(self, model_type='samvit_base_patch16.sa1b', num_classes=1, dr=8.0):
        super().__init__()
        droppath = 0.0
        model = timm.create_model(
                    model_type,
                    pretrained=True,
                    num_classes=num_classes,
                )
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        ##### get feature maps from the last four blocks.
        self.begin = 8
        self.len = 12
        
        ##### Begin inserting LoRA into MHSA module
        for t_layer_i, blk in enumerate(model.blocks):
            if t_layer_i < self.len:
                model.blocks[t_layer_i].attn.qkv = LoRA(blk.attn.qkv, 64)
                
        model.neck = nn.Identity() # remove original decoders 
        self.sam = model
        del model
        
        #### define the decoder
        self.num_classes = num_classes
        self.decoder = SAM_Decoder(768, 64, num_classes=num_classes, downsample_rate=dr)
        
    def forward_ppn(self, x):
        '''
        forward for the patch embedding, position embedding and pre-norm
        '''
        self.batch, self.depth = x.shape[0], x.shape[-1]
        x = x.expand(-1, 3, -1, -1, -1)
        x = desequence(x)
        x = self.sam.patch_embed(x)
        if self.sam.pos_embed is not None:
            x = x + resample_abs_pos_embed_nhwc(self.sam.pos_embed, x.shape[1:3])
        x = self.sam.pos_drop(x)
        x = self.sam.patch_drop(x)
        x = self.sam.norm_pre(x)
        return x
        
    def forward_encoder(self, x):
        res = []
        for i in range(self.len):
            x = self.sam.blocks[i](x)
            if i >= self.begin:
                res.append(x.permute(0, 3, 1, 2))
        return res
    
    def forward_decoder(self, x_lis):
        output = self.decoder(x_lis, self.batch, self.depth)
        return output
    
    def forward(self, x):
        x = self.forward_ppn(x)
        features = self.forward_encoder(x)
        output = self.forward_decoder(features)
        return output

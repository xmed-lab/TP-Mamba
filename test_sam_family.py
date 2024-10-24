from networks.sam.sam_base import SAM_Base
import torch

# SAM_Base: frozen the pre-trained SAM and set the decoder learnable
sam_base = SAM_Base(num_classes=2)
x = torch.randn(1, 1, 96, 96, 96) # batch, channel (default 1 for CT), Height, Width, Depth
y = sam_base(x)
print(y.size())
print(sam_base)
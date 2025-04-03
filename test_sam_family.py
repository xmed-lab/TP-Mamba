from networks.sam.sam_base import SAM_Base
from networks.sam.sam_ms import SAM_MS
import torch

# SAM_Base: frozen the pre-trained SAM and set the decoder learnable
sam_ms = SAM_MS(num_classes=2, dr=16.0)
x = torch.randn(1, 1, 96, 96, 96) # batch, channel (default 1 for CT), Height, Width, Depth
y = sam_ms(x)
print(y.size())
#print(sam_base)
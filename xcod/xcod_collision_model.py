import os
import torch
import torch.nn as nn
from base_models import PointTransformerV3


class XCoD(PointTransformerV3):
    def __init__(self, pretrained=False, model_dir=None, **kwargs):
        super().__init__(in_channels=4,
                         enable_flash=True, 
                         upcast_attention=False,
                         upcast_softmax=False,
                         **kwargs)
        # convert point representation to logits
        self.seg_head = nn.Linear(self.dec_dims, 2)

        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url("https://huggingface.co/prabinrath/xmop/resolve/main/xcod.pth", model_dir)
            self.load_state_dict(state_dict['model'])
        
    def forward(self, data_dict):
        point = super().forward(data_dict)
        point.feat = self.seg_head(point.feat)
        return point

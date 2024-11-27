"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

from modules.VisionTransformer import VisionTransformer



def Vit3D():
    model = VisionTransformer(img_size=64,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              in_c=1,
                              num_classes=2)
    return model


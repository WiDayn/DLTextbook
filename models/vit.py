# -*- coding: utf-8 -*-
# @时间: 2024-11-26
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 通用Vit模块
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
# 使用示例:
from unittest.mock import patch

from monai.networks.nets import ViT
from torch import nn

from modules.VisionTransformer import VisionTransformer


class Vit(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = ViT(
        in_channels=1,
        img_size=(64, 64, 64),
        patch_size=(16, 16, 16),
        classification=True,
    )

    def forward(self, x):
        x, hidden_states_out = self.model(x)
        return x


def Vit3D():
    model = VisionTransformer(img_size=64,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              in_c=1,
                              num_classes=2)
    return model


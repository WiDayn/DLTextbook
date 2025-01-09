# -*- coding: utf-8 -*-
# @时间: 2024-11-26
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: Vit模型
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
# 使用示例:
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer, SwinUNETR
from monai.utils import ensure_tuple_rep
import torch.nn.functional as F

from modules.VisionTransformer import VisionTransformer

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        # 使用全局平均池化
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # 输出维度为 (batch_size, in_channels, 1, 1, 1)

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)  # 最后一层是分类层

    def forward(self, x):
        x = self.global_pool(x)  # 输出形状：[batch_size, 384, 1, 1, 1]
        x = x.view(x.size(0), -1)  # 输出形状：[batch_size, 384]
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class swinTrans_model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SwinUNETR(
            img_size = [64, 64, 64],
            in_channels = 1,
            out_channels = 14,
            feature_size = 48,
        )
        self.model.load_state_dict(torch.load("./models/pretrain/monai_swin.pt"))
        self.classifier = Classifier(input_dim=768, num_classes=2)

    def forward(self, x, classify = True):
        hidden_states_out = self.model.swinViT(x, True)
        if classify:
            return self.classifier(hidden_states_out[4])
        return hidden_states_out[4]

def SwinTrans():
    model = swinTrans_model()
    return model


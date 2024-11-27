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


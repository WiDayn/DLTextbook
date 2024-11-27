# -*- coding: utf-8 -*-
# @时间: 2024-11-26
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 适用于CAL的Loss函数，注意已经包含了CE Loss不需要额外添加
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码): https://arxiv.org/abs/2108.08728
# 修改历史:
# 使用示例:
import torch.nn.functional as F
from torch import nn


class CAL_LOSS(nn.Module):
    def __init__(self):
        super(CAL_LOSS, self).__init__()
        self.EPSILON = 1e-6  # Small constant for stability

    # Cross-entropy loss
    def cross_entropy_loss(self, pred, target):
        return F.cross_entropy(pred, target)

    def forward(self, outputs, y):
        """
        Compute the total loss for training, without augmentation and cropping.
        """
        [y_pred_raw, y_pred_aux] = outputs
        # Loss computation without augmentation
        loss_raw = self.cross_entropy_loss(y_pred_raw, y)
        loss_aux = self.cross_entropy_loss(y_pred_aux, y)

        # Combine losses with weights
        batch_loss = loss_raw / 3. + \
                     loss_aux * 3. / 3.

        return batch_loss
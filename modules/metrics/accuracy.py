# -*- coding: utf-8 -*-
# @时间: 2024-11-27
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 计算ACC
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
#
# 使用示例:
import torch


def accuracy(logits, targets):
    preds = torch.softmax(logits, dim=1)
    _, predicted = preds.max(1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)
# -*- coding: utf-8 -*-
# @时间: 2024-11-26
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 计算F1
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
# 使用示例:
import torch
from sklearn.metrics import f1_score as sklearn_f1

def f1_score(logits, targets):
    preds = torch.softmax(logits, dim=1)
    _, predicted = preds.max(1)
    return sklearn_f1(targets.cpu(), predicted.cpu(), average='macro')
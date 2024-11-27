# -*- coding: utf-8 -*-
# @时间: 2024-11-26
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 计算AUC
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
# 使用示例:
import torch
from sklearn.metrics import roc_auc_score


def auc_score(logits, targets):
    preds = torch.softmax(logits, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()

    # 如果是二分类问题，直接计算 AUC
    if preds.shape[1] == 2:
        preds = preds[:, 1]  # 仅使用正类的概率
        auc = roc_auc_score(targets, preds)
    else:
        # 如果是多分类，使用 `average='macro'` 对每个类别计算 one-vs-rest AUC
        auc = roc_auc_score(targets, preds, average='macro', multi_class='ovr')

    return auc
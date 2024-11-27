# -*- coding: utf-8 -*-
# @时间: 2024-11-26
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 计算P R
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
# 使用示例:
import torch


def recall(logits, targets):
    preds = torch.softmax(logits, dim=1)
    _, predicted = preds.max(1)

    # 计算 TP, FP 和 FN
    TP = ((predicted == 1) & (targets == 1)).sum().item()  # True Positives
    TN = ((predicted == 0) & (targets == 0)).sum().item()
    FN = ((predicted == 0) & (targets == 1)).sum().item()  # False Negatives
    FP = ((predicted == 1) & (targets == 0)).sum().item()  # True Positives

    # 计算精确率和召回率
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    print("TP TN FP FN ", TP, TN, FP, FN)

    return recall

def precision(logits, targets):
    preds = torch.softmax(logits, dim=1)
    _, predicted = preds.max(1)

    # 计算 TP, FP 和 FN
    TP = ((predicted == 1) & (targets == 1)).sum().item()  # True Positives
    FP = ((predicted == 1) & (targets == 0)).sum().item()  # False Positives

    # 计算精确率和召回率
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    return precision
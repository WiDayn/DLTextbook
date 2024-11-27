# -*- coding: utf-8 -*-
# @时间: 2024-11-26
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 标准CrossTransformer模块
# 输入: 两个模态的特征向量，输出CA后的两个特征向量
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
# 使用示例:
# 假设现在存在两个特征向量A,B
# 输入顺序应该为A, B, B, B, A, A
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionWithTransformer(nn.Module):
    def __init__(self, query_dim, key_value_dim):
        super(CrossAttentionWithTransformer, self).__init__()

        self.attention_1 = nn.MultiheadAttention(embed_dim=query_dim, num_heads=5)
        self.attention_2 = nn.MultiheadAttention(embed_dim=query_dim, num_heads=5)

        self.query_linear_1 = nn.Linear(query_dim, query_dim)
        self.key_linear_1 = nn.Linear(key_value_dim, query_dim)
        self.value_linear_1 = nn.Linear(key_value_dim, query_dim)

        self.query_linear_2 = nn.Linear(query_dim, query_dim)
        self.key_linear_2 = nn.Linear(key_value_dim, query_dim)
        self.value_linear_2 = nn.Linear(key_value_dim, query_dim)

        self.layer_norm_1 = nn.LayerNorm(query_dim)
        self.layer_norm_2 = nn.LayerNorm(query_dim)

    def forward(self, query_1, key_1, value_1, query_2, key_2, value_2):
        query_1 = self.query_linear_1(query_1).unsqueeze(0)  # Shape: (1, batch_size, query_dim)
        key_1 = self.key_linear_1(key_1).unsqueeze(0)  # Shape: (1, batch_size, key_value_dim)
        value_1 = self.value_linear_1(value_1).unsqueeze(0)  # Shape: (1, batch_size, key_value_dim)

        query_2 = self.query_linear_2(query_2).unsqueeze(0)  # Shape: (1, batch_size, query_dim)
        key_2 = self.key_linear_2(key_2).unsqueeze(0)  # Shape: (1, batch_size, key_value_dim)
        value_2 = self.value_linear_2(value_2).unsqueeze(0)  # Shape: (1, batch_size, key_value_dim)

        output_1, _ = self.attention_1(query_1, key_1, value_1)
        output_1 = self.layer_norm_1(output_1.squeeze(0))

        output_2, _ = self.attention_2(query_2, key_2, value_2)
        output_2 = self.layer_norm_2(output_2.squeeze(0))

        return output_1, output_2

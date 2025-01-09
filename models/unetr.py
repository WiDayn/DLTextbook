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
from modules.networks.unetr import UNETR


def UNET_R():
    model = UNETR(
        in_channels=1,  # 输入通道数，假设是1，适用于单通道图像（比如灰度图像）
        out_channels=2,  # 输出通道数，假设是2，适用于二分类问题
        img_size=(64, 64, 64),  # 图像尺寸，3D图像的大小，例如 (128, 128, 128)
        feature_size=16,  # 特征图大小
        hidden_size=768,  # Transformer中的隐藏层大小
        mlp_dim=3072,  # MLP的维度
        num_heads=12,  # 注意力头数
        pos_embed="perceptron",  # 位置编码方式
        norm_name="instance",  # 正则化方式，采用实例归一化
        conv_block=False,  # 是否使用卷积块
        res_block=True,  # 是否使用残差块
        dropout_rate=0.0  # Dropout比率
    )
    return model


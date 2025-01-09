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
from monai.networks.nets import UNet, ResNet, ResNetBlock, resnet34, resnet50, resnet101, resnet152


def RESNET18():
    model = ResNet(
        block=ResNetBlock,               # 使用标准 ResNet 块
        layers=[2, 2, 2, 2],             # 每个阶段的 ResNet 块数（标准 ResNet18 配置）
        block_inplanes=[64, 128, 256, 512],  # 每个阶段的通道数
        spatial_dims=3,                  # 3D 输入
        n_input_channels=1,              # 单通道输入（灰度图像）
        conv1_t_size=7,                  # 第一层卷积核大小
        conv1_t_stride=2,                # 第一层卷积步幅
        no_max_pool=False,               # 是否移除最大池化
        shortcut_type="B",               # 残差连接类型
        widen_factor=1.0,                # 宽度因子，默认为1
        num_classes=2,                   # 二分类
        feed_forward=True,               # 启用前馈
        bias_downsample=True             # 兼容性选项
    )
    return model

def RESNET34():
    model = resnet34(
        spatial_dims = 3,
        num_classes=2,
        n_input_channels=1,
    )
    return model

def RESNET50():
    model = resnet50(
        spatial_dims = 3,
        num_classes=2,
        n_input_channels=1,
    )
    return model

def RESNET101():
    model = resnet101(
        spatial_dims = 3,
        num_classes=2,
        n_input_channels=1,
    )
    return model

def RESNET152():
    model = resnet152(
        spatial_dims = 3,
        num_classes=2,
        n_input_channels=1,
    )
    return model
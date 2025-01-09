from typing import Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT
class UNETRClassification(nn.Module):
    """
    UNETR for 3D image classification based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,  # 分类类别数
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: 输入通道数
            num_classes: 类别数（分类任务的输出数）
            img_size: 输入图像的尺寸
            feature_size: 特征大小
            hidden_size: Transformer隐藏层维度
            mlp_dim: Transformer中MLP层的维度
            num_heads: Transformer中的头数
            pos_embed: 位置编码方式
            norm_name: 归一化方式
            dropout_rate: Dropout率
        """

        super().__init__()

        # 确保Dropout率有效
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        # 确保hidden_size能够被num_heads整除
        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        # 位置编码类型
        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)  # Patch大小
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=False,  # 取消分类模式，保留特征输出
            dropout_rate=dropout_rate,
        )

        # 编码器层
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
        )

        # 分类层
        self.fc = nn.Linear(hidden_size * self.feat_size[0] * self.feat_size[1] * self.feat_size[2], num_classes)

    def forward(self, x_in):
        # 获取Transformer输出
        x, hidden_states_out = self.vit(x_in)

        # 使用encoder4的输出作为分类输入
        enc4 = self.encoder4(x_in)  # 使用encoder4处理输入

        # 将Transformer输出展平
        x_flat = x.view(x.size(0), -1)

        # 分类输出
        out = self.fc(x_flat)
        return out

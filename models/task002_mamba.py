# -*- coding: utf-8 -*-
# @时间: 2024-11-26
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: Mamba模型
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
# 使用示例:
import torch
import torch.nn as nn
from mamba_ssm import Mamba

def count_parameters(model):
    """
    计算PyTorch模型的参数量
    Args:
        model (nn.Module): 需要计算参数量的模型
    Returns:
        int: 模型的总参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000 / 1000


# Patch Embedding模块
class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, patch_size=(4, 4, 4)):
        super(PatchEmbedding3D, self).__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        return x


# MambaLayer定义
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.channel_token = channel_token  ## whether to use channel as tokens

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)

        return out

    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)

        return out + x


# 构建纯Mamba的网络结构
class PureMambaNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1000, embed_dim=64, patch_size=(8, 8, 8), num_layers=4):
        super(PureMambaNet, self).__init__()

        # Patch Embedding层
        self.patch_embed = PatchEmbedding3D(in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size)

        # 多个Mamba层
        self.mamba_layers = nn.ModuleList([
            MambaLayer(dim=embed_dim, d_state=16, d_conv=4, expand=2, channel_token=False)
            for _ in range(num_layers)
        ])

        # 全局平均池化和分类层
        self.relu = nn.LeakyReLU()
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)

        for layer in self.mamba_layers:
            x = layer(x)

        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

def FullResMamba():
    model = PureMambaNet(in_channels=1, num_classes=2, embed_dim=32, patch_size=(4, 4, 4), num_layers=8)
    return model
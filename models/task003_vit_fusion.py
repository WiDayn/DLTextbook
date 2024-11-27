# -*- coding: utf-8 -*-
# @时间: 2024-11-26
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 组学，深度学习特征融合模型
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
# 使用示例:
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from cc3d import reshape

from modules.CrossAttentionWithTransformer import CrossAttentionWithTransformer
from modules.VisionTransformer import VisionTransformer
from pytorch_grad_cam import GradCAM
from torch.nn import Conv3d
from torchvision.models.inception import BasicConv2d


class GenomicsEncoder(nn.Module):
    def __init__(self):
        super(GenomicsEncoder, self).__init__()
        self.fc1 = nn.Linear(3832, 1024)  # 组学数据有100个特征
        self.fc2 = nn.Linear(1024, 768)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 分类器
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)  # 最后一层是分类层

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def reshape_transform(tensor, d=4, w=4, h=4):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    d, w, h, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(3, 4).transpose(2, 3).transpose(1, 2)
    return result


EPSILON = 1e-8  # 防止计算中出现零


# Bilinear Attention Pooling (BAP) for 3D images
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        """
        初始化 BAP 模块。
        :param pool: 'GAP' 表示全局平均池化，'GMP' 表示全局最大池化
        """
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP'], "pool 参数必须是 'GAP' 或 'GMP'"

        # 选择池化方法
        if pool == 'GAP':
            self.pool = None  # 没有池化（全局平均池化）
        else:
            self.pool = nn.AdaptiveMaxPool3d(1)  # 3D 最大池化

    def forward(self, features, attentions):
        """
        前向传播：使用注意力图对特征图进行加权池化。
        :param features: 输入特征图，形状为 (B, C, D, H, W)
        :param attentions: 输入注意力图，形状为 (B, M, D, H, W)
        :return: 特征矩阵和反事实特征矩阵
        """
        B, C, D, H, W = features.size()  # 获取特征图的形状
        _, M, AD, AH, AW = attentions.size()  # 获取注意力图的形状

        # 如果注意力图的空间维度与特征图不匹配，则进行上采样
        if AD != D or AH != H or AW != W:
            attentions = F.interpolate(attentions, size=(D, H, W), mode='trilinear', align_corners=False)

        # 计算加权特征矩阵：首先进行爱因斯坦求和，结合特征图和注意力图
        if self.pool is None:
            # 如果是 GAP，则进行加权求和并平均池化
            feature_matrix = (torch.einsum('imdhw,indhw->imn', (attentions, features)) / float(D * H * W)).view(B, -1)
        else:
            # 使用最大池化
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)  # 对加权特征图进行池化
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)  # 将每个池化后的特征拼接起来

        # sign-sqrt 操作：取符号并对每个元素进行平方根变换
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # 对特征矩阵进行 L2 归一化
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        # 如果是训练模式，生成反事实特征矩阵
        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)  # 随机生成假的注意力图
        else:
            fake_att = torch.ones_like(attentions)  # 测试时使用统一的注意力权重

        # 使用假的注意力图进行加权特征计算
        counterfactual_feature = (torch.einsum('imdhw,indhw->imn', (fake_att, features)) / float(D * H * W)).view(B, -1)

        # 对反事实特征进行 sign-sqrt 和归一化
        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + EPSILON)
        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)

        return feature_matrix, counterfactual_feature

class JointEmbeddingModelWithCrossAttentionTransformer(nn.Module):
    def __init__(self, num_classes):
        super(JointEmbeddingModelWithCrossAttentionTransformer, self).__init__()
        self.genomics_embedding = []
        self.image_embedding = []
        self.image_encoder = VisionTransformer(img_size=64,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              in_c=1,
                              num_classes=0)
        self.num_features = 768
        self.image_fc = nn.Linear(self.num_features, 25)
        self.genomics_encoder = GenomicsEncoder()
        self.cross_attention = CrossAttentionWithTransformer(query_dim=25, key_value_dim=25)
        self.classifier = Classifier(input_dim=50, num_classes=num_classes)

    def forward(self, inputs, save_for_grad_cam=False):
        # image, genomics = inputs[0], inputs[1]
        image, genomics = inputs[:, 0, :, :, :].unsqueeze(1), inputs[:, 1:26, 0, 0, 0]
        image_feature = self.image_encoder(image, save_for_grad_cam, return_feature_maps=False) # torch.Size([32, 65, 768])
        self.image_embedding = self.image_fc(image_feature)
        # genomics_embedding = self.genomics_encoder(genomics)
        self.genomics_embedding = genomics

        # grayscale_cam = self.cam(input_tensor=inputs, targets=None)  # 输出形状 [B, D, H, W]
        # rand_cam = self.cam(input_tensor=torch.zeros_like(inputs), targets=None)  # 输出形状 [B, D, H, W]

        attention_output_1, attention_output_2 = self.cross_attention(
            self.image_embedding, self.genomics_embedding, self.genomics_embedding,
            self.genomics_embedding, self.image_embedding, self.image_embedding
        )

        combined_embedding = torch.cat((attention_output_1, attention_output_2), dim=1)

        # 传递给分类器
        output = self.classifier(combined_embedding)

        return output

def FusionModel():
    model = JointEmbeddingModelWithCrossAttentionTransformer(2)
    return model


# -*- coding: utf-8 -*-
# @时间: 2024-11-26
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 标准测试函数
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
# 使用示例:
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from datetime import datetime


def final_evaluate(model, dataloader, metrics, save_visuals=False, visuals_dir=None, title="", save_to_excel=True,
                   excel_file="_feature_gt.xlsx"):
    """
    对模型进行最终评估，支持多个指标，并可生成可视化内容（如 ROC 曲线）。
    :param model: 训练完成的模型
    :param dataloader: 验证数据加载器
    :param metrics: 评估指标字典
    :param save_visuals: 是否保存可视化文件
    :param visuals_dir: 可视化文件保存目录
    :param save_to_excel: 是否导出预测结果和真实值到 Excel
    :param excel_file: Excel 文件保存路径
    :return: 返回包含评估结果的字典
    """
    print(f"--------------------{title} Begin--------------------")
    excel_file = visuals_dir + "/" +title.replace(" ", "_") + excel_file
    model.eval()
    metric_results = {metric_name: 0 for metric_name in metrics}

    feature_tensors = []
    ground_truth_tensors = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            feature_tensors.append(outputs)
            ground_truth_tensors.append(targets)

    # 将所有feature tensors 拼接成一个大的tensor
    all_features = torch.cat([tensor.cpu() for tensor in feature_tensors], dim=0)
    # 将所有ground truth tensors 拼接成一个大的tensor
    all_ground_truth = torch.cat([tensor.cpu() for tensor in ground_truth_tensors], dim=0)

    # 确保特征数量和标签数量匹配
    assert all_features.size(0) == all_ground_truth.size(0), "特征数量和标签数量不匹配"

    # 将特征转换为numpy数组
    features_np = all_features.numpy()
    # 将标签转换为numpy数组
    labels_np = all_ground_truth.numpy()

    # 创建一个DataFrame
    # 为特征命名，例如 feature_1, feature_2, ..., feature_n
    feature_columns = [f'VD_feature_{i + 1}' for i in range(features_np.shape[1])]
    df_features = pd.DataFrame(features_np, columns=feature_columns)
    # 添加GroundTruth列
    df_features['GroundTruth'] = labels_np

    # 导出为Excel文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{'/'.join(excel_file.split('/')[:-1])}_{timestamp}{excel_file.split('_')[-1]}"
    df_features.to_excel(output_path, index=False)

    print(f"数据已成功保存到 {output_path}")

    return {}

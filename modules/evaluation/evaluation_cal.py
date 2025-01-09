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
                   excel_file="_results.xlsx"):
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

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            all_outputs.append(outputs)
            all_targets.append(targets)

    # first_outputs = [x[0] for x in all_outputs]
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets)

    # 计算各个指标
    for metric_name, metric_fn in metrics.items():
        metric_results[metric_name] = metric_fn(all_outputs, all_targets)

    # 打印评估结果
    print("Final Evaluation Results:")
    for metric_name, value in metric_results.items():
        print(f"{metric_name}: {value:.4f}")

    # 如果需要生成可视化文件
    if save_visuals and visuals_dir:
        os.makedirs(visuals_dir, exist_ok=True)
        # 生成 ROC 曲线
        if all_outputs.shape[1] == 2:  # 二分类情况
            fpr, tpr, _ = roc_curve(all_targets.cpu().numpy(), torch.softmax(all_outputs, dim=1)[:, 1].cpu().numpy())
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=1.2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC')
            plt.legend(loc="lower right")
            roc_path = os.path.join(visuals_dir, "roc_curve.png")
            plt.savefig(roc_path)
            print(f"ROC curve saved to {roc_path}")
        else:
            print("ROC visualization is only available for binary classification.")

    # 如果需要导出预测结果和真实值到Excel
    if save_to_excel:
        # 将输出和目标转换为 numpy 数组
        all_name = dataloader.dataset.image_files
        all_outputs_np = torch.softmax(all_outputs.cpu(), dim=1).numpy()
        all_targets_np = all_targets.cpu().numpy()

        # 对于多分类，获取最大概率对应的类别作为预测结果
        if all_outputs_np.ndim == 2:
            predicted_labels = all_outputs_np.argmax(axis=1)
        else:  # 二分类
            predicted_labels = (all_outputs_np > 0.5).astype(int)

        # 创建 DataFrame 并保存为 Excel 文件
        df = pd.DataFrame({
            'Data Name': all_name,
            'True Labels': all_targets_np.flatten(),
            'Predicted Labels': predicted_labels.flatten(),
            'Predicted Probabilities': all_outputs_np[:, 1] if all_outputs_np.shape[1] == 2 else None
        })

        # 如果是二分类且包含概率信息
        if all_outputs_np.shape[1] == 2:
            df['Predicted Probabilities'] = all_outputs_np[:, 1]  # 只保存类别 1 的概率

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file_with_timestamp = f"{os.path.splitext(excel_file)[0]}_{timestamp}{os.path.splitext(excel_file)[1]}"

        # 保存 Excel 文件
        os.makedirs(os.path.dirname(visuals_dir), exist_ok=True)
        df.to_excel(excel_file_with_timestamp, index=False)
        print(f"Evaluation results exported to {excel_file_with_timestamp}")

    return metric_results

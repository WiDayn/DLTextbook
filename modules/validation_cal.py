# -*- coding: utf-8 -*-
# @时间: 2024-11-26
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 用于具有CAL模块的模型的验证函数
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
# 使用示例:
import torch

def validate(model, val_loader, criterion, metrics):
    model.eval()
    val_loss = 0
    metric_results = {metric: 0 for metric in metrics}

    sum_outputs = []
    sum_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # 计算每个指标
            sum_outputs.append(outputs)
            sum_targets.append(targets)

    first_outputs = [x[0] for x in sum_outputs]
    sum_outputs = torch.cat(first_outputs, dim=0)
    sum_targets = torch.cat(sum_targets)

    for metric_name, metric_fn in metrics.items():
        metric_results[metric_name] += metric_fn(sum_outputs, sum_targets)

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # 创建结果字典并添加平均损失和所有指标
    results = {'loss': avg_val_loss}
    for metric_name, metric_value in metric_results.items():
        results[metric_name] = metric_value
        print(f"{metric_name}: {metric_value:.4f}")

    model.train()  # 切换回训练模式
    return results

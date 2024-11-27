# -*- coding: utf-8 -*-
# @时间: 2024-11-26
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 从config中加载内容并保存为字典
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
# 使用示例:
import re
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # 从文件名推测任务名称（例如 `task1_config.yaml` -> `task1`）
    task_name = re.search(r'task\d+', config_path)
    if task_name:
        task_name = task_name.group()
    else:
        task_name = "default"  # 如果没有匹配到任务名称，则使用默认配置

    # 设置默认组件路径
    config.setdefault('model', f"models.{task_name}_model.{task_name.capitalize()}Model")
    config.setdefault('dataset', f"datasets.{task_name}_dataset.{task_name.capitalize()}Dataset")
    config.setdefault('dataloader', f"data.dataloaders.{task_name}_dataloader.{task_name.capitalize()}Dataloader")
    config.setdefault('loss_function', "torch.nn.CrossEntropyLoss")

    # 默认优化器和学习率衰减策略
    config.setdefault('optimizer', {
        'type': "torch.optim.Adam",
        'params': {'lr': config.get('learning_rate', 0.001)}
    })

    config.setdefault('scheduler', None)  # 默认不使用学习率衰减

    return config
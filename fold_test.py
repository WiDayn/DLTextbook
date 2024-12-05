# -*- coding: utf-8 -*-
# @时间: YYYY-MM-DD HH:MM:SS
# @作者: 作者名
# @邮箱: 邮箱地址
#
# 整体描述:
# 输入: 描述输入参数和格式
# 输出: 描述输出数据和格式
# 限制性条件: 例如输入数据范围、函数执行环境等
# 算法/数据来源(论文/代码): 相关论文名称、链接或代码来源
# 修改历史: 列出修改日期、修改人及修改内容
# - 2024.11.12 曾强 调整xxx
#
# 使用示例:
# 1. 示例代码，展示如何调用主要函数
# 2. 运行结果展示


import os

import torch

from utils.config_loader import load_config
from utils.instance_loader import get_instance, get_metric_function

def test(config, model_file, device, sequence, fold):
    # 加载数据预处理函数
    train_transform = get_instance(config['preprocessing']['train_transform'])
    val_transform = get_instance(config['preprocessing']['val_transform'])

    # 应用预处理并创建数据集, 加载数据集和数据加载器
    test_dataset = get_instance(config['dataset'], config['test_dataset_dir'], transform=val_transform, set_type="test", device=device, sequence=sequence, fold=fold)

    test_loader = get_instance(config['dataloader'], test_dataset, batch_size=config['batch_size']).get_loader()

    # 加载模型、损失函数、优化器和学习率衰减策略
    model = get_instance(config['model']).to(device)

    model.load_state_dict(torch.load(model_file))

    print("Model Name:", config['model'].split('.')[-1])
    print("Model File:", model_file)

    # 最终模型评估
    all_test_metric_results = []
    if "evaluation" in config:
        eval_metrics = {metric.split('.')[-1]: get_metric_function(metric) for metric in
                        config["evaluation"]["evaluation_metrics"]}
        save_visuals = config["evaluation"].get("save_visuals", False)
        visuals_dir = config["evaluation"].get("visuals_dir", "./visuals")
        all_test_metric_results.append(get_instance(config["evaluation"]["function"], model, test_loader, eval_metrics,
                     save_visuals=save_visuals, visuals_dir=visuals_dir, title=model_file.split('/')[-1].split('.')[0] + " Test Result"))

    return [all_test_metric_results]

def fold_Test(device, save_dir, config, sequence, best_metric_name):
    test_avg = {}

    for fold in range(5):
        # 调用验证函数
        model_path = os.path.join(save_dir, f"best_{config['model'].split('.')[-1]}_{sequence}_fold{fold + 1}_{best_metric_name}.pth")
        all_test_metric_results, = test(config, model_path, device, sequence, fold)  # 进行验证
        for result in all_test_metric_results:
            for metric_name, value in result.items():
                if metric_name not in test_avg:
                    test_avg[metric_name] = 0
                test_avg[metric_name] += value

    for k in test_avg:
        test_avg[k] /= 5

    print("Test Avg", test_avg)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # config = load_config("configs/task003_rxafusion.yaml")
    # sequence = "AD"
    config = load_config("configs/task003_rxafusion_cal.yaml")
    sequence = "V40"
    save_dir = config.get("save_dir", "./checkpoints/" + config['task_name'])
    best_metric_name = config.get("best_metric_name", "accuracy")
    fold_Test(device, save_dir, config, sequence, best_metric_name)
    # val(config,'checkpoints/task002/Vit_best_model_auc_score.pth', device, sequence)

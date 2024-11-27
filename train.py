# -*- coding: utf-8 -*-
# @时间: 2024-11-27
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 用于执行训练（非五折）的主函数
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
#
# 使用示例:
import os
import sys
import time

import argparse

import torch
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard
from nnunetv2.utilities.network_initialization import InitWeights_He
from utils.config_loader import load_config
from utils.instance_loader import get_instance, get_metric_function

from utils.setup_seed import setup_seed


def setup_logging(log_dir):
    # 设置日志文件名为时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(log_dir, f"log_{timestamp}.txt")

    # 设置标准输出流同时输出到控制台和文件
    class DualOutput:
        def __init__(self, console, file):
            self.console = console
            self.file = file

        def write(self, message):
            self.console.write(message)
            self.file.write(message)

        def flush(self):
            self.console.flush()
            self.file.flush()

    os.makedirs(log_dir, exist_ok=True)
    log_file = open(log_file, 'w')
    dual_output = DualOutput(sys.stdout, log_file)
    sys.stdout = dual_output  # 重定向print输出
    return log_file


def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration Override")

    # 添加命令行参数
    parser.add_argument('--config', type=str, help='Override config path')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--gpu_id', type=int, help='Override GPU ID to use')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--save_frequency', type=int, help='Override model save frequency')
    parser.add_argument('--model', type=str, help='Override model class')
    parser.add_argument('--sequence', type=str, help='Override sequence name')

    # 解析命令行参数
    args = parser.parse_args()
    return args

def train(config, args):
    # 设置命令行参数覆盖config中的值
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.gpu_id is not None:
        config['gpu_id'] = args.gpu_id
    if args.epochs:
        config['epochs'] = args.epochs
    if args.save_frequency:
        config['save_frequency'] = args.save_frequency
    if args.sequence:
        config['sequence'] = args.sequence

    # 设置日志文件夹
    log_file = setup_logging(config.get("log_dir", "./logs"))

    # 是否启用 TensorBoard
    tensorboard_enabled = config.get("tensorboard", {}).get("enabled", False)
    writer = None
    if tensorboard_enabled:
        log_dir = config["tensorboard"].get("log_dir", "./logs")
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard is enabled. Logs will be saved to: {log_dir}")

    # sequence
    sequence = config['sequence']
    print(f"Now train sequence {sequence}")

    # 设置使用的GPU
    gpu_id = config.get('gpu_id', 0)  # 默认为使用GPU 0
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    print(f"Using GPU {gpu_id}.")

    # 模型保存设置
    save_frequency = config.get("save_frequency", 5)
    save_dir = config.get("save_dir", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # 要监控的最佳指标名称
    best_metric_name = config.get("best_metric_name", "accuracy")
    best_metric_value = float('-inf')  # 初始化为负无穷大

    # 加载数据预处理函数
    train_transform = get_instance(config['preprocessing']['train_transform'])
    val_transform = get_instance(config['preprocessing']['val_transform'])

    # 应用预处理并创建数据集, 加载数据集和数据加载器
    train_dataset = get_instance(config['dataset'], config['train_dataset_dir'], transform=train_transform,
                                     set_type="train", fold=-1, sequence=config['sequence'], device=device)
    val_dataset = get_instance(config['dataset'], config['val_dataset_dir'], transform=val_transform,
                                     set_type="val", fold=-1, sequence=config['sequence'], device=device)

    train_loader = get_instance(config['dataloader'], train_dataset, batch_size=config['batch_size']).get_loader()
    val_loader = get_instance(config['dataloader'], val_dataset, batch_size=config['batch_size']).get_loader()

    # 加载模型、损失函数、优化器和学习率衰减策略
    model = get_instance(config['model']).to(device)
    model = model.apply(InitWeights_He(1e-2))
    criterion = get_instance(config['loss_function']).to(device)
    optimizer = get_instance(config['optimizer']['type'], model.parameters(), **config['optimizer']['params'])

    scheduler = None
    if config['scheduler']:
        scheduler = get_instance(config['scheduler']['type'], optimizer, **config['scheduler']['params'])

    # 加载验证指标
    metrics = {metric.split('.')[-1]: get_metric_function(metric) for metric in config['validation']['metrics']}

    # 训练与验证过程
    model.train()
    validation_frequency = config.get('validation_frequency', 1)

    best_metric_value = float('-inf')  # 初始化为负无穷大

    for epoch in range(config['epochs']):
        # 获取并打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        epoch_loss = 0

        # 训练
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{config['epochs']}, Training Loss: {avg_epoch_loss:.4f}, Current Learning Rate: {current_lr:.6f}")

        # 将训练损失和学习率记录到 TensorBoard
        if writer:
            writer.add_scalar(f"Loss/Train", avg_epoch_loss)
            writer.add_scalar(f"Learning_Rate", current_lr)

        # 每隔 `validation_frequency` 轮进行一次验证
        if (epoch + 1) % validation_frequency == 0:
            val_metrics = get_instance(config['validation']['function'], model, val_loader, criterion, metrics)
            if writer:
                writer.add_scalar(f"Loss/Val", val_metrics['loss'])
                for metric_name, metric_value in val_metrics.items():
                    if metric_name != 'loss':
                        writer.add_scalar(f"Metric/{metric_name}", metric_value)

            # 检查当前的验证指标是否超过历史最佳
            current_metric_value = val_metrics.get(best_metric_name, None)
            if current_metric_value is not None and current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                best_model_path = os.path.join(save_dir,
                                               f"best_{config['model'].split('.')[-1]}_{config['sequence']}_{best_metric_name}.pth")
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"New best model saved at {best_model_path} with {best_metric_name}: {best_metric_value:.4f}")

        # 更新学习率
        if scheduler:
            scheduler.step()

        # 每隔 `save_frequency` 个 epoch 保存一次模型
        if (epoch + 1) % save_frequency == 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            save_path = os.path.join(save_dir,
                                     f"best_{config['model'].split('.')[-1]}_{config['sequence']}_epoch{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved at {save_path}")

    # 关闭 TensorBoard 写入
    if writer:
        writer.close()

if __name__ == "__main__":
    args = parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config("configs/task003_rxafusion_cal.yaml")
    seed = config.get('seed', -1)
    setup_seed(seed)
    train(config, args)

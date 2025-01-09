# -*- coding: utf-8 -*-
# @时间: 2024-11-26
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 批量执行训练任务
# 输入: 描述输入参数和格式
# 输出: 描述输出数据和格式
# 限制性条件: 例如输入数据范围、函数执行环境等
# 算法/数据来源(论文/代码): 相关论文名称、链接或代码来源
# 修改历史:
# 使用示例:

import subprocess
import time
import os
import signal


def get_gpu_memory():
    """
    使用 nvidia-smi 获取所有 GPU 的显存占用信息。
    返回一个字典，key 为 GPU ID，value 为显存占用（单位：MB）。
    """
    gpu_memory = {}
    # 获取 nvidia-smi 输出，包含显存使用信息
    command = "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits"
    result = subprocess.check_output(command, shell=True).decode("utf-8").strip().splitlines()

    for line in result:
        index, memory_used, memory_total = map(int, line.split(","))
        gpu_memory[index] = memory_used / 1024  # 转换为 GB
    return gpu_memory


def get_available_gpu(gpu_ids, min_memory=1.0):
    """
    获取指定 GPU ID 列表中显存占用少于 min_memory (GB) 的 GPU ID。
    """
    gpu_memory = get_gpu_memory()
    available_gpus = [gpu_id for gpu_id in gpu_ids if gpu_memory.get(gpu_id, 0) < min_memory]
    return available_gpus


def run_train_command(command, gpu_id):
    """
    运行训练命令，指定使用的 GPU，并返回子进程对象。
    """
    full_command = f"CUDA_VISIBLE_DEVICES={gpu_id} {command}"
    print(f"Running command: {full_command}")
    # 使用 subprocess.Popen 启动训练任务，以便可以后续监控
    process = subprocess.Popen(full_command, shell=True, preexec_fn=os.setsid)
    return process


def wait_for_gpu(min_memory=1.0, gpu_ids_to_check=[0, 1]):
    """
    检查并等待至少一个 GPU 空闲（显存小于指定值）。
    """
    while True:
        available_gpus = get_available_gpu(gpu_ids_to_check, min_memory)
        if available_gpus:
            print(f"Available GPUs found: {available_gpus}")
            return available_gpus
        # print("No available GPUs. Waiting...")
        time.sleep(5)  # 每 5 秒检查一次

def terminate_processes(processes):
    """
    终止所有运行的子进程。
    """
    print("\nTerminating all running processes...")
    for process in processes:
        try:
            # 使用 os.killpg 终止整个进程组
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            print(f"Terminated process {process.pid}")
        except Exception as e:
            print(f"Failed to terminate process {process.pid}: {e}")

def main():
    # 需要执行的训练命令列表（可以根据需求修改）
    train_commands = [
        "python train.py --sequence A40",
        "python train.py --sequence A70",
        "python train.py --sequence AD",
        "python train.py --sequence AZ",
        "python train.py --sequence V40",
        "python train.py --sequence V70",
        "python train.py --sequence VD",
        "python train.py --sequence VZ",
    ]

    start_time = time.time()

    # 指定你希望检查的 GPU ID 列表
    gpu_ids_to_check = [1, 2, 3]  # 假设你要检查 GPU 0 和 GPU 1

    # 持续循环直到所有训练命令执行完毕
    processes = []  # 用于保存所有的子进程

    try:
        while train_commands:
            # 获取空闲 GPU
            available_gpus = wait_for_gpu(min_memory=1.0, gpu_ids_to_check=gpu_ids_to_check)

            # 执行训练任务
            command = train_commands.pop(0)  # 取出队列中的第一个训练命令
            gpu_id = available_gpus[0]  # 选择一个空闲的 GPU 执行任务

            # 运行任务并返回子进程对象
            process = run_train_command(command, gpu_id)
            processes.append(process)  # 保存子进程

            # 等待短暂的时间，避免过快地进行下一个任务
            time.sleep(30)

        # 等待所有任务执行完毕
        for process in processes:
            process.wait()  # 阻塞，直到子进程完成
            print("A training task has finished.")

        print("All tasks completed.")
    except KeyboardInterrupt:
        # 捕获 Ctrl+C 中断
        print("\nKeyboardInterrupt detected! Cleaning up...")
        terminate_processes(processes)

    finally:
        # 确保无论如何都清理子进程
        terminate_processes(processes)
        print("Spend time", time.time() - start_time)


if __name__ == "__main__":
    main()

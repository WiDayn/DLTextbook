# -*- coding: utf-8 -*-
# @时间: 2024-11-26
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 示例自定义Scheduler
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
# 使用示例:
# 假设现在存在两个特征向量A,B
# 输入顺序应该为A, B, B, B, A, A
class WarmupLR:
    def __init__(self, optimizer, num_warm) -> None:
        self.optimizer = optimizer
        self.num_warm = num_warm
        self.lr = [group['lr'] for group in self.optimizer.param_groups]
        self.num_step = 0

    def __compute(self, lr) -> float:
        return lr * min(self.num_step ** (-0.5), self.num_step * self.num_warm ** (-1.5))

    def step(self) -> None:
        self.num_step += 1
        lr = [self.__compute(lr) for lr in self.lr]
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lr[i]

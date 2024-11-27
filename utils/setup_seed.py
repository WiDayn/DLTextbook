# -*- coding: utf-8 -*-
# @时间: 2024-11-27
# @作者: 曾强
# @邮箱: skv@live.com
#
# 整体描述: 为torch，random,np设置种子
# 输入:
# 输出:
# 限制性条件:
# 算法/数据来源(论文/代码):
# 修改历史:
# 使用示例:
import random
import time

import numpy as np
import torch


def setup_seed(seed):
    if seed == -1:
        seed = random.seed(time.time())
        print(f"No seed be setting in config, try to random one by current time: {seed}")
    print(f"Your seed is {seed}.")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
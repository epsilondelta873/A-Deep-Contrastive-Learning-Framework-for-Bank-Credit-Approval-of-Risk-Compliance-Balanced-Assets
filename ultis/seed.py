# -*- coding: utf-8 -*-
'''
@File    :   seed.py
@Time    :   2026/02/05 21:53:30
@Author  :   chensy 
@Desc    :   随机种子设置工具，用于保证实验可复现性
'''

import random
import numpy as np
import torch
import os


def set_seed(seed=123):
    """设置全局随机种子以保证实验可复现性。
    
    该函数会固定以下随机性来源：
    1. Python 内置 random 模块
    2. NumPy 随机数生成器
    3. PyTorch CPU 随机数生成器
    4. PyTorch CUDA 随机数生成器
    5. CUDA 确定性算法
    
    Args:
        seed (int): 随机种子值，默认 42
        
    Example:
        >>> from ultis.seed import set_seed
        >>> set_seed(42)  # 固定随机种子为 42
        >>> # 之后的所有随机操作将是可复现的
    """
    # 1. 设置 Python 内置 random 模块的种子
    random.seed(seed)
    
    # 2. 设置 NumPy 随机数生成器的种子
    np.random.seed(seed)
    
    # 3. 设置 PyTorch CPU 随机数生成器的种子
    torch.manual_seed(seed)
    
    # 4. 设置 PyTorch CUDA 随机数生成器的种子（如果使用 GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 对于多 GPU 情况
    
    # 5. 设置 CUDA 的确定性行为
    # 注意：这可能会降低性能，但能保证完全可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 6. 设置环境变量（对于某些 CUDA 操作）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ 随机种子已设置为: {seed}")
    print(f"  - Python random seed: {seed}")
    print(f"  - NumPy seed: {seed}")
    print(f"  - PyTorch seed: {seed}")
    if torch.cuda.is_available():
        print(f"  - CUDA seed: {seed}")
        print(f"  - CUDNN deterministic: True")
        print(f"  - CUDNN benchmark: False")

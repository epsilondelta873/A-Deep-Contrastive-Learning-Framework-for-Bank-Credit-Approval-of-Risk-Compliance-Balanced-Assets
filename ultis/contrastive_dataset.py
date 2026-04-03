# -*- coding: utf-8 -*-
'''
@File    :   contrastive_dataset.py
@Time    :   2026/02/08 19:30:00
@Author  :   chensy 
@Desc    :   对比学习数据集类，支持混合有标签和无标签数据
'''

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .augmentation import TabularAugmentation


class ContrastiveDataset(Dataset):
    """
    对比学习数据集
    
    每个样本返回两个随机增强的视图，用于对比学习训练。
    支持同时加载多个数据文件（有标签和无标签数据）。
    """
    
    def __init__(self, file_paths, feature_names=None, augmentation_config=None):
        """
        初始化对比学习数据集
        
        Args:
            file_paths (list): Excel 文件路径列表，例如 ['train_accepted_woe.xlsx', 'train_rejected_woe.xlsx']
            feature_names (list, optional): 要使用的特征列名列表
            augmentation_config (dict, optional): 数据增强配置
        """
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.feature_names = feature_names
        
        # 初始化数据增强器
        if augmentation_config is None:
            augmentation_config = {'noise_level': 0.1, 'drop_prob': 0.15, 'use_both': True}
        self.augmenter = TabularAugmentation(**augmentation_config)
        
        # 加载所有数据文件并合并
        self._load_data()
    
    def _load_data(self):
        """加载并合并多个数据文件"""
        dfs = []
        
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            print(f"Loading data from {file_path}...")
            df = pd.read_excel(file_path)
            dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError("No valid data files found!")
        
        # 合并所有数据
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"Total samples loaded: {len(self.df)}")
        
        # 确定特征列
        if self.feature_names is None:
            # 排除非特征列
            exclude_cols = ['y', 'profit', 'score', 'id', 'w', 'pname', 'cost']
            self.feature_names = [c for c in self.df.columns if c not in exclude_cols]
        
        # 过滤非数值列
        numeric_features = []
        for col in self.feature_names:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_features.append(col)
        
        if len(numeric_features) < len(self.feature_names):
            filtered_count = len(self.feature_names) - len(numeric_features)
            print(f"Warning: Filtered {filtered_count} non-numeric features")
        
        self.feature_names = numeric_features
        
        if len(self.feature_names) == 0:
            raise ValueError("No numeric features found after filtering!")
        
        print(f"Using {len(self.feature_names)} features for contrastive learning")
        
        # 提取特征为 NumPy 数组
        self.X = self.df[self.feature_names].values.astype(np.float32)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        获取单个样本的两个增强视图
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (view1, view2) 两个增强视图的张量
        """
        # 获取原始特征
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        
        # 生成两个不同的增强视图
        view1 = self.augmenter(x)
        view2 = self.augmenter(x)
        
        return view1, view2


def get_contrastive_dataloader(data_dir, file_names, feature_names=None, 
                               batch_size=256, augmentation_config=None):
    """
    创建对比学习 DataLoader
    
    Args:
        data_dir (str): 数据目录路径
        file_names (list): 数据文件名列表，例如 ['train_accepted_woe.xlsx', 'train_rejected_woe.xlsx']
        feature_names (list, optional): 特征列表
        batch_size (int): 批次大小，建议 256-512
        augmentation_config (dict, optional): 数据增强配置
        
    Returns:
        DataLoader: 对比学习数据加载器
    """
    # 构建完整文件路径
    file_paths = [os.path.join(data_dir, fname) for fname in file_names]
    
    # 创建数据集
    dataset = ContrastiveDataset(
        file_paths=file_paths,
        feature_names=feature_names,
        augmentation_config=augmentation_config
    )
    
    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 表格数据通常不需要多进程
        drop_last=True   # 丢弃最后一个不完整的 batch（对对比学习很重要）
    )
    
    return dataloader


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    print("--- 对比学习数据集测试 ---\n")
    
    # 配置
    data_dir = "../data/processed"
    file_names = ["train_accepted_woe.xlsx", "train_rejected_woe.xlsx"]
    batch_size = 32
    
    # 数据增强配置
    aug_config = {
        'noise_level': 0.1,
        'drop_prob': 0.15,
        'use_both': True
    }
    
    try:
        # 创建 DataLoader
        dataloader = get_contrastive_dataloader(
            data_dir=data_dir,
            file_names=file_names,
            batch_size=batch_size,
            augmentation_config=aug_config
        )
        
        print(f"\n✓ DataLoader 创建成功！")
        print(f"  总 batch 数: {len(dataloader)}")
        print(f"  批次大小: {batch_size}")
        
        # 获取一个 batch 测试
        view1, view2 = next(iter(dataloader))
        print(f"\n第一个 batch:")
        print(f"  View1 shape: {view1.shape}")
        print(f"  View2 shape: {view2.shape}")
        print(f"  特征维度: {view1.shape[1]}")
        
        # 验证两个视图不同
        diff = torch.abs(view1 - view2).mean().item()
        print(f"\n两个视图的平均差异: {diff:.4f}")
        
        if diff > 0.01:
            print("✓ 数据增强工作正常！")
        else:
            print("⚠ 警告：两个视图过于相似，可能需要增大增强强度")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        print("请确保数据文件存在于 data/processed 目录")

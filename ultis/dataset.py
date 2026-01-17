# -*- coding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2026/01/17 21:25:30
@Author  :   chensy 
@Desc    :   数据集加载模块，提供 PyTorch DataLoader 封装
'''

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def select_features(iv_path, top_n=None, data_type='woe'):
    """根据 IV 值选择特征。
    
    Args:
        iv_path (str): IV 结果 Excel 文件的路径。
        top_n (int, optional): 选择排名前 N 的特征数量。如果为 None，则选择全部特征。
        data_type (str): 'woe' 或 'raw'。决定变量命名规则。
                        'woe' 会在变量名后添加 '_woe' 后缀。
                        'raw' 保持原始变量名。
    
    Returns:
        list: 选中的特征名称列表。
        
    Raises:
        FileNotFoundError: IV 文件不存在。
        ValueError: IV 文件缺少必要列或 data_type 参数错误。
    """
    if not os.path.exists(iv_path):
        raise FileNotFoundError(f"IV file not found at {iv_path}")
        
    df_iv = pd.read_excel(iv_path)
    
    # 确保按 IV 值降序排列
    if 'iv' in df_iv.columns:
        df_iv = df_iv.sort_values(by='iv', ascending=False)
    
    # 获取基础变量名
    # 假设存在 'variable' 列（由 calculate_iv.py 输出）
    if 'variable' not in df_iv.columns:
        raise ValueError("IV file must contain 'variable' column.")
        
    all_vars = df_iv['variable'].tolist()
    
    # 选择前 N 个特征或全部特征
    if top_n is not None and isinstance(top_n, int):
        selected_vars = all_vars[:top_n]
    else:
        selected_vars = all_vars
        
    # 根据 data_type 处理变量名
    if data_type == 'woe':
        final_features = [f"{var}_woe" for var in selected_vars]
    elif data_type == 'raw':
        final_features = selected_vars
    else:
        raise ValueError("data_type must be 'woe' or 'raw'")
        
    return final_features


class CreditDataset(Dataset):
    """
    信贷数据集类
    
    该类将 Excel 数据文件封装为 PyTorch Dataset，用于创建 DataLoader。
    """
    
    def __init__(self, file_path, feature_names=None):
        """初始化信用数据集。
        
        Args:
            file_path (str): Excel 文件的路径。
            feature_names (list, optional): 要使用的特征列名列表。
                                          如果为 None，则使用除 'y'、'profit' 和 'score' 外的所有列。
        
        Raises:
            FileNotFoundError: 数据文件不存在。
            ValueError: 目标列或利润列不存在，或特征列缺失。
        """
        self.file_path = file_path
        self.feature_names = feature_names
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")
            
        print(f"Loading data from {file_path}...")
        self.df = pd.read_excel(file_path)
        
        # 验证必须的列是否存在
        if 'y' not in self.df.columns:
            raise ValueError("Target column 'y' not found in dataset.")
        if 'profit' not in self.df.columns:
            raise ValueError("Profit column 'profit' not found in dataset.")
            
        # 如果未指定特征名，则使用除 y、profit 和 score 外的所有列
        if self.feature_names is None:
            exclude_cols = ['y', 'profit', 'score'] 
            self.feature_names = [c for c in self.df.columns if c not in exclude_cols]
        else:
            # 检查所有特征是否存在
            missing = [f for f in self.feature_names if f not in self.df.columns]
            if missing:
                raise ValueError(f"Missing features in dataset: {missing}")
                
        # 提取特征、标签和利润为 NumPy 数组
        self.X = self.df[self.feature_names].values.astype(np.float32)
        self.y = self.df['y'].values.astype(np.float32)
        self.profit = self.df['profit'].values.astype(np.float32)
        
    def __len__(self):
        """返回数据集样本数量。
        
        Returns:
            int: 数据集中的样本总数。
        """
        return len(self.df)
    
    def __getitem__(self, idx):
        """获取单个样本。
        
        Args:
            idx (int): 样本索引。
            
        Returns:
            tuple: (features, label, profit) 包含特征张量、标签张量和利润值的三元组。
        """
        return (
            torch.tensor(self.X[idx]), 
            torch.tensor(self.y[idx]), 
            torch.tensor(self.profit[idx])
        )


def get_dataloaders(data_dir=None, iv_path=None, batch_size=64, 
                    top_n_features=20, data_type='woe'):
    """为训练集、测试集和拒绝集创建 DataLoader。
    
    所有 DataLoader 统一返回 (features, y, profit) 三元组。
    
    Args:
        data_dir (str, optional): 包含处理后 Excel 文件的目录。
                                  如果为 None，则默认为相对于此文件的 '../data/processed'。
        iv_path (str, optional): IV 结果 Excel 文件的路径。
                                 如果为 None，则默认为 data_dir 中的 'iv_results.xlsx'。
        batch_size (int): 批次大小。
        top_n_features (int or None): 使用排名前 N 的 IV 特征数量。
        data_type (str): 'woe' 或 'raw'。
        
    Returns:
        dict: 包含 'train'、'test'、'reject' DataLoader 的字典。
              每个 DataLoader 返回 (features, y, profit) 三元组。
        
    Example:
        >>> loaders = get_dataloaders(batch_size=32, top_n_features=20)
        >>> for X_batch, y_batch, profit_batch in loaders['train']:
        >>>     # 训练逻辑（忽略 profit_batch）
        >>>     pass
    """
    # 0. 如果未提供路径，则自动检测
    if data_dir is None:
        # 假设项目结构：project/ultis/dataset.py -> project/data/processed
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_dir = os.path.join(project_root, 'data', 'processed')
        
    if iv_path is None:
        iv_path = os.path.join(data_dir, 'iv_results.xlsx')
        
    # 1. 选择特征
    feature_names = select_features(iv_path, top_n=top_n_features, data_type=data_type)
    print(f"Selected {len(feature_names)} features ({data_type} mode).")
    
    # 2. 根据 data_type 定义文件名
    # 命名规则：train_accepted_woe.xlsx / train_accepted_raw.xlsx
    suffix = f"_{data_type}.xlsx"
    
    train_path = os.path.join(data_dir, "train_accepted" + suffix)
    test_path = os.path.join(data_dir, "test" + suffix)
    reject_path = os.path.join(data_dir, "train_rejected" + suffix)
    
    data_loaders = {}
    
    # 3. 创建 DataLoader
    # 训练集 Loader
    if os.path.exists(train_path):
        train_dataset = CreditDataset(train_path, feature_names)
        data_loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        print(f"Warning: Train file not found: {train_path}")
        
    # 测试集 Loader
    if os.path.exists(test_path):
        test_dataset = CreditDataset(test_path, feature_names)
        data_loaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        print(f"Warning: Test file not found: {test_path}")

    # 拒绝集 Loader
    if os.path.exists(reject_path):
        reject_dataset = CreditDataset(reject_path, feature_names)
        data_loaders['reject'] = DataLoader(reject_dataset, batch_size=batch_size, shuffle=False)
    else:
         print(f"Warning: Reject file not found: {reject_path}")
         
    return data_loaders


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    """
    示例 1: 极简用法（自动推断路径）
    """
    print("--- 示例 1: 简单用法 ---")
    dataloaders = get_dataloaders(
        batch_size=32,
        top_n_features=20
    )
    
    if 'train' in dataloaders:
        features, labels, profits = next(iter(dataloaders['train']))
        print(f"Train Batch - Features shape: {features.shape}, Labels shape: {labels.shape}, Profits shape: {profits.shape}")

    if 'test' in dataloaders:
        test_features, test_labels, test_profits = next(iter(dataloaders['test']))
        print(f"Test Batch - Features shape: {test_features.shape}, Profits shape: {test_profits.shape}")

    if 'reject' in dataloaders:
        reject_features, reject_labels, reject_profits = next(iter(dataloaders['reject']))
        print(f"Reject Batch - Features shape: {reject_features.shape}, Profits shape: {reject_profits.shape}")

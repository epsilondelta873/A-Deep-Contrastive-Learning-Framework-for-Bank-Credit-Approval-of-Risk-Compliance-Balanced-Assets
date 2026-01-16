import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def select_features(iv_path, top_n=None, data_type='woe'):
    """
    Select features based on IV values.
    
    Args:
        iv_path (str): Path to the IV results excel file.
        top_n (int, optional): Number of top features to select. If None, select all.
        data_type (str): 'woe' or 'raw'. Determines variable naming convention.
                         'woe' appends '_woe' to variable names.
                         'raw' keeps original variable names.
    
    Returns:
        list: List of selected feature names.
    """
    if not os.path.exists(iv_path):
        raise FileNotFoundError(f"IV file not found at {iv_path}")
        
    df_iv = pd.read_excel(iv_path)
    
    # Ensure sorted by IV descending
    if 'iv' in df_iv.columns:
        df_iv = df_iv.sort_values(by='iv', ascending=False)
    
    # Get base variable names
    # Assuming 'variable' column exists (output from calculate_iv.py)
    if 'variable' not in df_iv.columns:
        raise ValueError("IV file must contain 'variable' column.")
        
    all_vars = df_iv['variable'].tolist()
    
    # Select top N or all
    if top_n is not None and isinstance(top_n, int):
        selected_vars = all_vars[:top_n]
    else:
        selected_vars = all_vars
        
    # Process variable names based on data_type
    if data_type == 'woe':
        final_features = [f"{var}_woe" for var in selected_vars]
    elif data_type == 'raw':
        final_features = selected_vars
    else:
        raise ValueError("data_type must be 'woe' or 'raw'")
        
    return final_features

class CreditDataset(Dataset):
    def __init__(self, file_path, feature_names=None, target_col='y'):
        """
        Args:
            file_path (str): Path to the excel file.
            feature_names (list, optional): List of feature column names to use. 
                                          If None, uses all columns except target and 'score'.
            target_col (str): Name of the target column. Default 'y'.
        """
        self.file_path = file_path
        self.feature_names = feature_names
        self.target_col = target_col
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")
            
        print(f"Loading data from {file_path}...")
        self.df = pd.read_excel(file_path)
        
        # Verify target column exists
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")
            
        # If feature_names is None, infer specific logic or use all except target
        if self.feature_names is None:
            exclude_cols = [self.target_col, 'score'] 
            self.feature_names = [c for c in self.df.columns if c not in exclude_cols]
        else:
            # Check if all features exist
            missing = [f for f in self.feature_names if f not in self.df.columns]
            if missing:
                raise ValueError(f"Missing features in dataset: {missing}")
                
        self.X = self.df[self.feature_names].values.astype(np.float32)
        self.y = self.df[self.target_col].values.astype(np.float32) # Assuming binary or regression target
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def get_dataloaders(data_dir=None, iv_path=None, batch_size=64, top_n_features=20, target_col='y', data_type='woe'):
    """
    Create DataLoaders for train, test, and reject datasets.
    
    Args:
        data_dir (str, optional): Directory containing the processed excel files. 
                                  If None, defaults to '../data/processed' relative to this file.
        iv_path (str, optional): Path to the IV results excel file. 
                                 If None, defaults to 'iv_results.xlsx' in data_dir.
        batch_size (int): Batch size.
        top_n_features (int or None): Number of top IV features to use.
        target_col (str): Target column name ('y' or 'profit').
        data_type (str): 'woe' or 'raw'.
        
    Returns:
        dict: Dictionary containing 'train', 'test', 'reject' DataLoaders.
    """
    # 0. Auto-detect paths if not provided
    if data_dir is None:
        # Assuming structure: project/ultis/dataset.py -> project/data/processed
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_dir = os.path.join(project_root, 'data', 'processed')
        
    if iv_path is None:
        iv_path = os.path.join(data_dir, 'iv_results.xlsx')
        
    # 1. Select features
    feature_names = select_features(iv_path, top_n=top_n_features, data_type=data_type)
    print(f"Selected {len(feature_names)} features ({data_type} mode).")
    
    # 2. Define file names based on data_type
    # Assuming naming convention: train_accepted_woe.xlsx / train_accepted_raw.xlsx
    suffix = f"_{data_type}.xlsx"
    
    train_path = os.path.join(data_dir, "train_accepted" + suffix)
    test_path = os.path.join(data_dir, "test" + suffix)
    reject_path = os.path.join(data_dir, "train_rejected" + suffix)
    
    data_loaders = {}
    
    # Train Loader
    if os.path.exists(train_path):
        train_dataset = CreditDataset(train_path, feature_names, target_col)
        data_loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        print(f"Warning: Train file not found: {train_path}")
        
    # Test Loader
    if os.path.exists(test_path):
        test_dataset = CreditDataset(test_path, feature_names, target_col)
        data_loaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        print(f"Warning: Test file not found: {test_path}")

    # Reject Loader
    if os.path.exists(reject_path):
        reject_dataset = CreditDataset(reject_path, feature_names, target_col)
        data_loaders['reject'] = DataLoader(reject_dataset, batch_size=batch_size, shuffle=False)
    else:
         print(f"Warning: Reject file not found: {reject_path}")
         
    return data_loaders

"""
# ==========================================
# 使用示例 (Usage Examples)
# ==========================================

if __name__ == "__main__":
    # 假设当前目录结构:
    # project/
    #   ultis/dataset.py
    #   data/processed/ (包含 train_accepted_woe.xlsx, iv_results.xlsx 等)
    
    # 示例 1: 极简用法 (自动寻找路径，默认配置)
    print("--- Example 1: Simple Usage ---")
    # 可以不传 path，函数会自动寻找 ../data/processed
    dataloaders = get_dataloaders(
        batch_size=32,
        top_n_features=20
    )
    
    if 'train' in dataloaders:
        features, labels = next(iter(dataloaders['train']))
        print(f"Train Batch - Features shape: {features.shape}, Labels shape: {labels.shape}")

    # 提取测试集和拒绝集数据示例
    if 'test' in dataloaders:
        test_features, test_labels = next(iter(dataloaders['test']))
        print(f"Test Batch - Features shape: {test_features.shape}")

    if 'reject' in dataloaders:
        reject_features, _ = next(iter(dataloaders['reject'])) # 拒绝集标签通常全为0或需要预测
        print(f"Reject Batch - Features shape: {reject_features.shape}")

    # 示例 2: 使用 Raw 数据，取所有变量，目标为 'profit' (假设 profit 列存在)
    # print("\\n--- Example 2: Raw Data, All Vars, Target='profit' ---")
    # try:
    #     dataloaders_raw = get_dataloaders(
    #         batch_size=16,
    #         top_n_features=None, # None 表示所有变量
    #         target_col='profit',
    #         data_type='raw'
    #     )
    #     if 'train' in dataloaders_raw:
    #         features, labels = next(iter(dataloaders_raw['train']))
    #         print(f"Train Batch (Raw) - Features shape: {features.shape}")
    # except Exception as e:
    #     print(f"Example 2 skipped or failed: {e}")
"""

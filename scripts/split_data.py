# -*- coding: utf-8 -*-
'''
@File    :   split_data.py
@Time    :   2026/01/17 21:12:15
@Author  :   chensy 
@Desc    :   数据集拆分：基于逻辑回归打分生成训练/测试/拒绝集
'''

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():
    """
    数据集拆分的主流程
    
    该函数执行以下步骤：
    1. 读取原始数据（raw）、WOE数据和IV结果
    2. 选择 Top 20 IV 变量
    3. 拆分训练集和测试集（70/30）
    4. 在训练集上构建逻辑回归模型并打分
    5. 根据分数将训练集分为接受集（前80%）和拒绝集（后20%）
    6. 导出所有数据集（raw 和 woe 两个版本）
    
    说明：
        - 拒绝集（Rejected）模拟信贷业务中被拒绝的客户样本
        - 接受集（Accepted）用于后续模型训练
        - 测试集（Test）用于最终模型评估
    
    输入文件：
        data/processed/p1_raw_data.xlsx - 原始数据
        data/processed/p1_woe_data.xlsx - WOE 转换后的数据
        data/processed/iv_results.xlsx - IV 值结果
        
    输出文件：
        data/processed/test_raw.xlsx - 测试集（原始值）
        data/processed/test_woe.xlsx - 测试集（WOE值）
        data/processed/train_accepted_raw.xlsx - 训练接受集（原始值）
        data/processed/train_accepted_woe.xlsx - 训练接受集（WOE值）
        data/processed/train_rejected_raw.xlsx - 训练拒绝集（原始值）
        data/processed/train_rejected_woe.xlsx - 训练拒绝集（WOE值）
    """
    # 步骤 1: 定义路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_data_path = os.path.join(project_root, 'data', 'processed', 'p1_raw_data.xlsx')
    woe_data_path = os.path.join(project_root, 'data', 'processed', 'p1_woe_data.xlsx')
    iv_results_path = os.path.join(project_root, 'data', 'processed', 'iv_results.xlsx')
    
    output_dir = os.path.join(project_root, 'data', 'processed')
    
    # 步骤 2: 读取数据
    print("正在加载数据...")
    df_raw = pd.read_excel(raw_data_path)
    df_woe = pd.read_excel(woe_data_path)
    df_iv = pd.read_excel(iv_results_path)
    
    # 验证数据一致性
    if len(df_raw) != len(df_woe):
        print("Error: Raw data 和 WoE data 行数不一致！")
        return

    # 步骤 3: 特征选择（基于 Top 20 IV）
    print("正在进行特征选择...")
    df_iv_sorted = df_iv.sort_values(by='iv', ascending=False)
    top_20_vars = df_iv_sorted.head(20)['variable'].tolist()
    
    # 映射到 WOE 变量名
    top_20_woe_vars = [f"{var}_woe" for var in top_20_vars]
    
    print(f"Top 20 IV 变量: {top_20_vars}")
    print(f"对应的 WOE 变量: {top_20_woe_vars}")
    
    # 验证列是否存在
    missing_cols = [col for col in top_20_woe_vars if col not in df_woe.columns]
    if missing_cols:
        print(f"Error: 以下WoE变量在数据集中未找到: {missing_cols}")
        return

    # 步骤 4: 拆分训练集和测试集（70/30）
    print("拆分训练集和测试集 (70/30)...")
    indices = np.arange(len(df_raw))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
    
    # 根据索引提取数据
    train_raw = df_raw.iloc[train_idx].copy()
    test_raw = df_raw.iloc[test_idx].copy()
    
    train_woe = df_woe.iloc[train_idx].copy()
    test_woe = df_woe.iloc[test_idx].copy()
    
    print(f"训练集大小: {len(train_raw)}, 测试集大小: {len(test_raw)}")
    
    # 步骤 5: 构建逻辑回归模型
    print("正在构建逻辑回归模型...")
    X_train = train_woe[top_20_woe_vars]
    y_train = train_woe['y']
    
    # 处理可能的缺失值
    if X_train.isnull().any().any():
        print("Warning: 训练数据中存在缺失值，正在使用均值填充...")
        X_train = X_train.fillna(X_train.mean())

    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # 步骤 6: 为训练集打分并生成拒绝集
    print("正在为训练集打分...")
    # 预测正类概率（违约概率）
    train_scores = clf.predict_proba(X_train)[:, 1]
    
    # 将分数添加到数据集
    train_woe['score'] = train_scores
    train_raw['score'] = train_scores
    
    # 按分数升序排列（分数低 -> 高）
    train_woe_sorted = train_woe.sort_values(by='score', ascending=True)
    train_raw_sorted = train_raw.sort_values(by='score', ascending=True)
    
    # 计算分割点（80% 位置）
    n_train = len(train_woe_sorted)
    cutoff_index = int(n_train * 0.8)
    
    # 前 80% -> 接受集（Accepted），后 20% -> 拒绝集（Rejected）
    train_accepted_woe = train_woe_sorted.iloc[:cutoff_index]
    train_rejected_woe = train_woe_sorted.iloc[cutoff_index:]
    
    train_accepted_raw = train_raw_sorted.iloc[:cutoff_index]
    train_rejected_raw = train_raw_sorted.iloc[cutoff_index:]
    
    print(f"接受集大小: {len(train_accepted_woe)} (前 80%)")
    print(f"拒绝集大小: {len(train_rejected_woe)} (后 20%)")
    
    # 步骤 7: 导出结果
    print("正在导出文件...")
    
    # 测试集
    test_raw.to_excel(os.path.join(output_dir, 'test_raw.xlsx'), index=False)
    test_woe.to_excel(os.path.join(output_dir, 'test_woe.xlsx'), index=False)
    
    # 训练接受集
    train_accepted_raw.to_excel(os.path.join(output_dir, 'train_accepted_raw.xlsx'), index=False)
    train_accepted_woe.to_excel(os.path.join(output_dir, 'train_accepted_woe.xlsx'), index=False)
    
    # 训练拒绝集
    train_rejected_raw.to_excel(os.path.join(output_dir, 'train_rejected_raw.xlsx'), index=False)
    train_rejected_woe.to_excel(os.path.join(output_dir, 'train_rejected_woe.xlsx'), index=False)
    
    print("所有文件导出完成！")


if __name__ == "__main__":
    main()

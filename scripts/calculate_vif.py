# -*- coding: utf-8 -*-
'''
@File    :   calculate_vif.py
@Time    :   2026/01/15 17:25:00
@Author  :   chensy 
@Desc    :   计算特征的方差膨胀因子（VIF）进行多重共线性分析
'''

import pandas as pd
import numpy as np
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def main():
    """
    计算 WOE 转换后特征的 VIF 值并合并到 IV 结果中
    
    该函数执行以下步骤：
    1. 读取 WOE 转换后的数据
    2. 筛选以 'x' 开头的特征变量
    3. 添加常数项后计算每个特征的 VIF 值
    4. 导出独立的 VIF 结果
    5. 将 VIF 结果合并到 IV 结果文件中（移除 _woe 后缀）
    
    说明：
        VIF (Variance Inflation Factor) 用于检测多重共线性。
        一般认为 VIF > 10 表示存在严重共线性。
    
    输入文件：
        data/processed/p1_woe_data.xlsx - WOE 转换后的数据
        data/processed/iv_results.xlsx - IV 值结果（用于合并）
        
    输出文件：
        data/processed/vif_results.xlsx - VIF 值排序结果
        data/processed/iv_results.xlsx - 更新后的 IV+VIF 合并结果
    """
    # 步骤 1: 定义路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_path = os.path.join(project_root, 'data', 'processed', 'p1_woe_data.xlsx')
    output_path = os.path.join(project_root, 'data', 'processed', 'vif_results.xlsx')

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    # 步骤 2: 读取数据
    print("正在读取数据...")
    df = pd.read_excel(input_path)
    print(f"数据读取完成，形状为: {df.shape}")

    # 步骤 3: 筛选特征变量
    # 说明：仅对 WOE 转换后的变量（以 'x' 开头）计算 VIF
    x_cols = [col for col in df.columns if col.startswith('x')]
    
    if not x_cols:
        print("Error: No variables starting with 'x' found in the dataset.")
        return
        
    print(f"正在计算 {len(x_cols)} 个变量的 VIF...")
    X = df[x_cols]
    
    # 步骤 4: 计算 VIF
    # 说明：添加常数项（截距）以正确计算 VIF
    X_with_const = add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    
    # 计算每个变量的 VIF
    # 注意：i+1 是因为 X_with_const 的第 0 列是常数项
    vif_values = []
    for i in range(X.shape[1]):
        vif = variance_inflation_factor(X_with_const.values, i + 1)
        vif_values.append(vif)
    
    vif_data["vif"] = vif_values

    # 步骤 5: 排序并导出 VIF 结果
    vif_results = vif_data.sort_values(by="vif", ascending=False)
    
    vif_results.to_excel(output_path, index=False)
    print(f"独立的 VIF 结果已保存至: {output_path}")

    # 步骤 6: 将 VIF 结果合并到 IV 结果中
    # 说明：为了方便特征选择，将 VIF 和 IV 合并在一个文件中
    iv_path = os.path.join(project_root, 'data', 'processed', 'iv_results.xlsx')
    if os.path.exists(iv_path):
        print("正在将 VIF 结果合并至 iv_results.xlsx (移除 _woe 后缀)...")
        df_iv = pd.read_excel(iv_path)
        
        # 预处理：去掉变量名末尾的 _woe 后缀以匹配 IV 结果
        vif_data_clean = vif_data.copy()
        vif_data_clean['variable'] = vif_data_clean['variable'].str.replace('_woe$', '', regex=True)
        
        # 删除旧的 vif 列（如果存在）以避免列名冲突
        if 'vif' in df_iv.columns:
            df_iv = df_iv.drop(columns=['vif'])
            
        # 基于 variable 列进行左连接
        df_merged = pd.merge(df_iv, vif_data_clean, on='variable', how='left')
        
        # 更新 IV 结果文件
        df_merged.to_excel(iv_path, index=False)
        print(f"合并后的结果已更新至: {iv_path}")
    else:
        print(f"Warning: iv_results.xlsx not found at {iv_path}, skipping merge.")

    print("VIF 计算与合并任务完成！")


if __name__ == "__main__":
    main()
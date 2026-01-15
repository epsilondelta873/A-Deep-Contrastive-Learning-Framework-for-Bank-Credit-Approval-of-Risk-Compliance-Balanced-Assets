# -*- coding: utf-8 -*-
'''
@File    :   calculate_vif.py
@Time    :   2026/01/15 17:25:00
@Author  :   chensy 
@Desc    :   计算转换woe后的各变量vif 进行共线性分析
'''

import pandas as pd
import numpy as np
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def main():
    # 1. 定义路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_path = os.path.join(project_root, 'data', 'processed', 'p1_woe_data.xlsx')
    output_path = os.path.join(project_root, 'data', 'processed', 'vif_results.xlsx')

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    # 2. 读取数据
    print("正在读取数据...")
    df = pd.read_excel(input_path)
    print(f"数据读取完成，形状为: {df.shape}")

    # 3. 筛选变量
    # 仅计算以 'x' 开头的变量 (这些是 WOE 转换后的变量)
    x_cols = [col for col in df.columns if col.startswith('x')]
    
    if not x_cols:
        print("Error: No variables starting with 'x' found in the dataset.")
        return
        
    print(f"正在计算 {len(x_cols)} 个变量的 VIF...")
    X = df[x_cols]
    
    # 4. 计算 VIF
    # 注意：计算 VIF 通常需要包含常数项（截距）
    X_with_const = add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    
    # 计算每一个变量的 VIF
    # 这里的 i+1 是因为 X_with_const 的第 0 列是 const
    vif_values = []
    for i in range(X.shape[1]):
        vif = variance_inflation_factor(X_with_const.values, i + 1)
        vif_values.append(vif)
    
    vif_data["vif"] = vif_values

    # 5. 排序并输出
    vif_results = vif_data.sort_values(by="vif", ascending=False)
    
    # 导出独立的 VIF 结果 (可选，保留作为备份)
    vif_results.to_excel(output_path, index=False)
    print(f"独立的 VIF 结果已保存至: {output_path}")

    # 6. 将 VIF 结果合并到 IV 结果中
    iv_path = os.path.join(project_root, 'data', 'processed', 'iv_results.xlsx')
    if os.path.exists(iv_path):
        print("正在将 VIF 结果合并至 iv_results.xlsx (移除 _woe 后缀)...")
        df_iv = pd.read_excel(iv_path)
        
        # 预处理 vif_data: 去掉变量名末尾的 _woe
        vif_data_clean = vif_data.copy()
        vif_data_clean['variable'] = vif_data_clean['variable'].str.replace('_woe$', '', regex=True)
        
        # 合并结果 (基于 variable 列)
        # 如果 iv_results 中已经有了 vif 列（通常是之前合并失败留下的），先删掉它，避免产生 vif_x, vif_y
        if 'vif' in df_iv.columns:
            df_iv = df_iv.drop(columns=['vif'])
            
        df_merged = pd.merge(df_iv, vif_data_clean, on='variable', how='left')
        # 重新保存合并后的 IV 结果
        df_merged.to_excel(iv_path, index=False)
        print(f"合并后的结果已更新至: {iv_path}")
    else:
        print(f"Warning: iv_results.xlsx not found at {iv_path}, skipping merge.")

    print("VIF 计算与合并任务完成！")

if __name__ == "__main__":
    main()
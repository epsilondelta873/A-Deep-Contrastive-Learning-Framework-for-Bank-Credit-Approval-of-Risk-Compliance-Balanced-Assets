# -*- coding: utf-8 -*-
'''
@File    :   calculate_iv.py
@Time    :   2026/01/15 16:42:00
@Author  :   chensy 
@Desc    :   计算特征的 IV 值并进行 WOE 转换（基于 scorecardpy）
'''

import pandas as pd
import numpy as np
import os
import scorecardpy as sc


def main():
    """
    计算特征 IV 值并进行 WOE 转换的主流程
    
    该函数执行以下步骤：
    1. 读取原始数据
    2. 处理缺失值（-1 -> NaN）
    3. 筛选以 'x' 开头的特征变量
    4. 使用 scorecardpy 进行自动分箱并计算 IV 值
    5. 导出 IV 结果（按 IV 值降序排列）
    6. 将原始特征值转换为 WOE 值
    7. 导出 WOE 转换后的数据
    
    输入文件：
        data/processed/p1_raw_data.xlsx
        
    输出文件：
        data/processed/iv_results.xlsx - IV 值排序结果
        data/processed/p1_woe_data.xlsx - WOE 转换后的数据
    """
    # 步骤 1: 定义路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_path = os.path.join(project_root, 'data', 'processed', 'p1_raw_data.xlsx')
    output_iv_path = os.path.join(project_root, 'data', 'processed', 'iv_results.xlsx')
    output_woe_path = os.path.join(project_root, 'data', 'processed', 'p1_woe_data.xlsx')

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    # 步骤 2: 读取数据
    print("正在读取数据...")
    df = pd.read_excel(input_path)
    print(f"数据读取完成，形状为: {df.shape}")

    # 步骤 3: 预处理 - 处理缺失值
    # 说明：将 -1 替换为 NaN，scorecardpy 会自动处理缺失值分箱
    print("正在处理缺失值 (-1 -> NaN)...")
    df.replace(-1, np.nan, inplace=True)

    # 筛选以 'x' 开头的特征变量和目标变量 'y'
    x_cols = [col for col in df.columns if col.startswith('x')]
    target_col = 'y'
    
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in dataset.")
        return
    
    df_selected = df[x_cols + [target_col]]
    print(f"筛选后的变量数量: {len(x_cols)} (总计 {df_selected.shape[1]} 列)")

    # 步骤 4: 自动分箱与 IV 计算
    # 说明：sc.woebin 会自动进行最优分箱并计算每个特征的 IV 值
    print("正在进行自动分箱和 IV 计算 (sc.woebin)...")
    bins = sc.woebin(df_selected, y=target_col)

    # 步骤 5: 提取并排序 IV 结果
    print("提取 IV 结果并排序...")
    iv_list = []
    for var, bin_df in bins.items():
        # 从分箱结果中提取 total_iv（每个变量的总 IV 值）
        total_iv = bin_df['total_iv'].iloc[0]
        iv_list.append({'variable': var, 'iv': total_iv})
    
    df_iv = pd.DataFrame(iv_list).sort_values(by='iv', ascending=False)
    
    # 导出 IV 结果
    df_iv.to_excel(output_iv_path, index=False)
    print(f"IV 结果已保存至: {output_iv_path}")

    # 步骤 6: WOE 转换
    # 说明：将原始特征值转换为对应的 WOE 值
    print("正在将原始数据转换为 WOE 值 (sc.woebin_ply)...")
    df_woe_core = sc.woebin_ply(df_selected, bins)
    
    # 步骤 7: 合并非特征列（如 id, pname 等）
    # 说明：保留原始数据中不参与 IV 计算的列
    other_cols = [col for col in df.columns if col not in df_selected.columns]
    if other_cols:
        print(f"合并原始列: {other_cols}")
        df_woe = pd.concat([df[other_cols], df_woe_core], axis=1)
    else:
        df_woe = df_woe_core

    # 导出 WOE 转换后的数据
    df_woe.to_excel(output_woe_path, index=False)
    print(f"WOE 转换后的数据已保存至: {output_woe_path}")

    print("任务处理完成！")


if __name__ == "__main__":
    main()
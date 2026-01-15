# -*- coding: utf-8 -*-
'''
@File    :   calculate_iv.py
@Time    :   2026/01/15 16:42:00
@Author  :   chensy 
@Desc    :   计算p1各变量的iv值并进行WOE转换，仅针对x开头的变量，使用scorecardpy
'''

import pandas as pd
import numpy as np
import os
import scorecardpy as sc

def main():
    # 1. 定义路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_path = os.path.join(project_root, 'data', 'processed', 'p1_raw_data.xlsx')
    output_iv_path = os.path.join(project_root, 'data', 'processed', 'iv_results.xlsx')
    output_woe_path = os.path.join(project_root, 'data', 'processed', 'p1_woe_data.xlsx')

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    # 2. 读取数据
    print("正在读取数据...")
    df = pd.read_excel(input_path)
    print(f"数据读取完成，形状为: {df.shape}")

    # 3. 预处理
    # 处理缺失值: -1 和 NaN 都视为缺失。
    # scorecardpy 会自动处理 NaN，我们将 -1 替换为 NaN
    print("正在处理缺失值 (-1 -> NaN)...")
    df.replace(-1, np.nan, inplace=True)

    # 筛选只包含 'x' 开头的变量和目标变量 'y'
    x_cols = [col for col in df.columns if col.startswith('x')]
    target_col = 'y'
    
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in dataset.")
        return
    
    df_selected = df[x_cols + [target_col]]
    print(f"筛选后的变量数量: {len(x_cols)} (总计 {df_selected.shape[1]} 列)")

    # 4. 使用 scorecardpy 计算 IV 和分箱
    print("正在进行自动分箱和 IV 计算 (sc.woebin)...")
    # sc.woebin 会计算 IV 值
    bins = sc.woebin(df_selected, y=target_col)

    # 5. 提取并排序 IV 结果
    print("提取 IV 结果并排序...")
    iv_list = []
    for var, bin_df in bins.items():
        # 获取该变量的 IV 值 (通常在 bin_df 的 'total_iv' 列或计算得到)
        # scorecardpy 的 bin_df 中每行都有 total_iv，我们取第一行即可
        total_iv = bin_df['total_iv'].iloc[0]
        iv_list.append({'variable': var, 'iv': total_iv})
    
    df_iv = pd.DataFrame(iv_list).sort_values(by='iv', ascending=False)
    
    # 导出 IV 结果
    df_iv.to_excel(output_iv_path, index=False)
    print(f"IV 结果已保存至: {output_iv_path}")

    # 6. 将原始值转换为 WOE
    print("正在将原始数据转换为 WOE 值 (sc.woebin_ply)...")
    df_woe_core = sc.woebin_ply(df_selected, bins)
    
    # 7. 合并保留原始的非 'x' 且非 'y' 的列 (如 id, pname 等)
    # 找出不在 df_selected 中但在原 df 中的列
    other_cols = [col for col in df.columns if col not in df_selected.columns]
    if other_cols:
        print(f"合并原始列: {other_cols}")
        # 通过索引合并，确保行对应关系
        df_woe = pd.concat([df[other_cols], df_woe_core], axis=1)
    else:
        df_woe = df_woe_core

    # 导出 WOE 数据
    df_woe.to_excel(output_woe_path, index=False)
    print(f"WOE 转换后的数据已保存至: {output_woe_path}")

    print("任务处理完成！")

if __name__ == "__main__":
    main()
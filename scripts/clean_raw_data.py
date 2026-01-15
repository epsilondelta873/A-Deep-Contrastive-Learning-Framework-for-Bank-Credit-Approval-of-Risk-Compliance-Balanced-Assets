# -*- coding: utf-8 -*-
'''
@File    :   clean_raw_data.py
@Time    :   2026/01/14 22:26:04
@Author  :   chensy 
@Desc    :   清洗原始数据train.xlsx test.xlsx后合并为总体数据集
'''

import pandas as pd
import os

def main():
    # 定义路径
    # 假设脚本位于 'scripts/' 目录，数据位于 '../data/raw/'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')

    train_path = os.path.join(raw_data_dir, 'train.xlsx')
    test_path = os.path.join(raw_data_dir, 'test.xlsx')

    # 检查文件是否存在
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Data files not found in {raw_data_dir}")
        return

    # 1. 读取数据
    print("Reading data...")
    try:
        df_train = pd.read_excel(train_path)
        df_test = pd.read_excel(test_path)
    except Exception as e:
        print(f"Error reading Excel files: {e}")
        return

    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")

    # 2. 合并数据
    print("Merging data...")
    df_merged = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Merged shape: {df_merged.shape}")

    # 3. 数据去重
    print("Removing duplicates...")
    df_dedup = df_merged.drop_duplicates()
    print(f"Shape after deduplication: {df_dedup.shape}")

    # 4. 筛选 pname='p1' 的数据
    if 'pname' in df_dedup.columns:
        print("Filtering for pname='p1'...")
        df_filtered = df_dedup[df_dedup['pname'] == 'p1']
        print(f"Final shape: {df_filtered.shape}")

        # 保存处理后的结果
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
        
        output_path = os.path.join(processed_data_dir, 'p1_raw_data.xlsx')
        df_filtered.to_excel(output_path, index=False)
        print(f"Saved processed data to: {output_path}")
    else:
        print("Error: Column 'pname' not found in the dataset.")

if __name__ == "__main__":
    main()
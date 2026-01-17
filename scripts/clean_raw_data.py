# -*- coding: utf-8 -*-
'''
@File    :   clean_raw_data.py
@Time    :   2026/01/14 22:26:04
@Author  :   chensy 
@Desc    :   清洗并合并原始数据集（train.xlsx 和 test.xlsx）
'''

import pandas as pd
import os


def main():
    """
    清洗并合并原始数据的主流程
    
    该函数执行以下步骤：
    1. 读取 train.xlsx 和 test.xlsx
    2. 合并两个数据集
    3. 去除重复数据
    4. 筛选 pname='p1' 的数据
    5. 保存处理后的结果
    
    输入文件：
        data/raw/train.xlsx - 训练集原始数据
        data/raw/test.xlsx - 测试集原始数据
        
    输出文件：
        data/processed/p1_raw_data.xlsx - 清洗后的 p1 数据集
    """
    # 步骤 1: 定义路径
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

    # 步骤 2: 读取数据
    print("正在读取数据...")
    try:
        df_train = pd.read_excel(train_path)
        df_test = pd.read_excel(test_path)
    except Exception as e:
        print(f"Error reading Excel files: {e}")
        return

    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")

    # 步骤 3: 合并数据
    print("正在合并数据...")
    df_merged = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Merged shape: {df_merged.shape}")

    # 步骤 4: 数据去重
    print("正在去除重复数据...")
    df_dedup = df_merged.drop_duplicates()
    print(f"Shape after deduplication: {df_dedup.shape}")

    # 步骤 5: 筛选特定项目数据
    # 说明：只保留 pname='p1' 的数据用于后续分析
    if 'pname' in df_dedup.columns:
        print("正在筛选 pname='p1' 的数据...")
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
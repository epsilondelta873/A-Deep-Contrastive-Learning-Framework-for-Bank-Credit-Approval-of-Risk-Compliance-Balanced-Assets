# -*- coding: utf-8 -*-
"""
@File    :   aggregate_metrics.py
@Author  :   chensy
@Desc    :   实验结果统计聚合脚本。
             读取 csv 汇总数据并以分组形式自动算数出 Mean 和 Standard Deviation (Std)，
             直接输出为新的汇总 CSV 文件。
"""

import argparse
import os
import pandas as pd
import numpy as np


CONFIG_ORDER = [
    'base_config.yaml',
    'resnet_config.yaml',
    'contrastive_resnet_config.yaml',
    'profit_resnet_config.yaml',
    'lightgbm_config.yaml',
    'xgboost_config.yaml',
    'hybrid_lgbm_config.yaml',
    'hybrid_xgb_config.yaml',
]

SPLIT_ORDER = {
    'Test': 0,
    'Reject': 1,
}


def main():
    parser = argparse.ArgumentParser(description="按实验分组计算各指标均值与标准差，并输出新的 CSV")
    parser.add_argument("--csv", type=str, default="experiment_summary.csv", help="数据追加累加记录表路径")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="汇总结果输出路径；默认在原文件名后追加 _aggregated.csv"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: 找不到聚合的数据流入口文件 {args.csv}")
        return
        
    df = pd.read_csv(args.csv)
    
    if len(df) == 0:
        print("未提取到有效日志数据，为空报表表单。")
        return

    # 根据实验标识列，确保各类架构互相独立分组不会被错算
    group_cols = ['Algorithm', 'Config', 'Split']
    target_metrics = ['AUC', 'Total_Profit']

    # 强制转为数值，空字符串自动变为 NaN
    for metric in target_metrics:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')

    result_rows = []

    for name, group in df.groupby(group_cols, dropna=False):
        script, config, split = name
        seeds = group['Seed'].dropna().tolist() if 'Seed' in group.columns else []
        try:
            seeds = sorted(seeds, key=lambda x: int(x))
        except (ValueError, TypeError):
            seeds = sorted(map(str, seeds))

        row = {
            'Algorithm': script,
            'Config': config,
            'Split': split,
            'N': len(group),
            'Seeds': ",".join(map(str, seeds))
        }

        for metric in target_metrics:
            if metric not in group.columns or group[metric].isna().all():
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
                continue

            values = group[metric].dropna().values.astype(float)
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=0))

            if "Profit" in metric:
                row[f'{metric}_mean'] = round(mean_val, 2)
                row[f'{metric}_std'] = round(std_val, 2)
            else:
                row[f'{metric}_mean'] = round(mean_val, 6)
                row[f'{metric}_std'] = round(std_val, 6)

        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)
    config_order_map = {config_name: idx for idx, config_name in enumerate(CONFIG_ORDER)}
    result_df['_config_order'] = result_df['Config'].map(config_order_map).fillna(len(CONFIG_ORDER))
    result_df['_split_order'] = result_df['Split'].map(SPLIT_ORDER).fillna(len(SPLIT_ORDER))
    result_df = result_df.sort_values(
        by=['_config_order', '_split_order', 'Algorithm', 'Config', 'Split']
    ).reset_index(drop=True)
    result_df = result_df.drop(columns=['_config_order', '_split_order'])

    if args.output is not None:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.csv)
        output_path = f"{base}_aggregated{ext or '.csv'}"

    result_df.to_csv(output_path, index=False, encoding='utf-8')


if __name__ == "__main__":
    main()

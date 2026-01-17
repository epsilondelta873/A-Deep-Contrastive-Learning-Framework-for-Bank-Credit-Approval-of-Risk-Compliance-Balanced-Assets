# -*- coding: utf-8 -*-
'''
@File    :   evaluate.py
@Time    :   2026/01/17 21:57:22
@Author  :   chensy 
@Desc    :   模型评估脚本
            支持二分类和回归模型的评估：
            1. 二分类模型：计算 AUC、准确率等分类指标 + 利润指标
            2. 回归模型：计算利润指标
'''

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, 
    precision_score, recall_score
)
from ultis.dataset import get_dataloaders
from models import get_model
from configs import load_config, merge_args_with_config, Config


def calculate_classification_metrics(y_true, y_pred_prob, threshold=0.5):
    """计算二分类指标。
    
    Args:
        y_true (np.ndarray): 真实标签，shape (n_samples,)。
        y_pred_prob (np.ndarray): 预测概率，shape (n_samples,)。
        threshold (float): 分类阈值，默认 0.5。
    
    Returns:
        dict: 包含 AUC、准确率、精确率、召回率、F1 值的字典。
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    return {
        "AUC": roc_auc_score(y_true, y_pred_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }


def calculate_profit_metrics(y_pred, profits, model_type='classification', top_percent=0.3):
    """计算利润指标。
    
    根据模型类型选择不同的排序策略：
    - 二分类：预测概率越低越好（低违约概率 = 优质客户），升序排序
    - 回归：预测值越高越好（高profit预测 = 优质客户），降序排序
    
    Args:
        y_pred (np.ndarray): 预测值/概率，shape (n_samples,)。
        profits (np.ndarray): 实际利润值，shape (n_samples,)。
        model_type (str): 模型类型，'classification' 或 'regression'。
        top_percent (float): 选择排名前 N% 的样本，默认 0.3（30%）。
    
    Returns:
        dict: 包含总利润、平均利润和选中样本数的字典。
    """
    n_samples = len(y_pred)
    n_selected = int(n_samples * top_percent)
    
    # 根据模型类型确定排序方式
    if model_type == 'classification':
        # 二分类：按概率升序排序（概率低 = 优质客户）
        sorted_indices = np.argsort(y_pred)
    else:  # regression
        # 回归：按预测值降序排序（预测值高 = 优质客户）
        sorted_indices = np.argsort(y_pred)[::-1]
    
    # 选择前 N% 的样本
    top_indices = sorted_indices[:n_selected]
    selected_profits = profits[top_indices]
    
    # 计算利润指标
    total_profit = selected_profits.sum()
    avg_profit = selected_profits.mean() if n_selected > 0 else 0.0
    
    return {
        "Total_Profit": float(total_profit),
        "Avg_Profit": float(avg_profit),
        "Selected_Count": n_selected,
        "Total_Samples": n_samples
    }


def main():
    """主评估流程。"""
    # 1. 参数解析
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="配置文件路径，默认为 configs/base_config.yaml"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=None,
        help="模型名称（如 baseline），会覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="模型文件路径"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=['classification', 'regression'],
        default=None,
        help="模型类型：classification（二分类）或 regression（回归），会覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=None,
        help="数据目录路径，会覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--iv_path", 
        type=str, 
        default=None,
        help="IV 结果文件路径，会覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="评估结果输出文件路径，会覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="批次大小，会覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--top_n_features", 
        type=int, 
        default=None,
        help="选择 IV 值最高的前 N 个特征，会覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=None,
        help="选择排名前 N% 的样本计算利润，会覆盖配置文件中的设置"
    )
    
    args = parser.parse_args()
    
    # 0. 加载配置文件并合并命令行参数
    print(f"Loading configuration from {args.config}...")
    try:
        config = load_config(args.config)
        config = merge_args_with_config(config, args)
    except FileNotFoundError:
        print(f"Warning: Config file {args.config} not found, using command-line arguments only.")
        # 如果配置文件不存在，使用默认配置
        config = Config({
            'model': {'name': args.model_name or 'baseline'},
            'data': {
                'data_dir': args.data_dir,
                'iv_path': args.iv_path,
                'batch_size': args.batch_size or 64,
                'top_n_features': args.top_n_features or 20
            },
            'evaluation': {
                'model_type': args.model_type or 'classification',
                'top_percent': args.top_percent or 0.3,
                'output_file': args.output_file or 'evaluation_results.csv'
            }
        })
    
    # 打印当前使用的配置
    print("\n" + "="*50)
    print("当前配置:")
    print(f"  模型: {config.model.name}")
    print(f"  模型类型: {config.evaluation.model_type}")
    print(f"  批次大小: {config.data.batch_size}")
    print(f"  特征数量: {config.data.top_n_features}")
    print(f"  Top百分比: {config.evaluation.top_percent}")
    print(f"  输出文件: {config.evaluation.output_file}")
    print("="*50 + "\n")
    
    # 2. 加载数据
    print(f"Loading data...")
    dataloaders = get_dataloaders(
        data_dir=config.data.data_dir,
        iv_path=config.data.iv_path,
        batch_size=config.data.batch_size,
        top_n_features=config.data.top_n_features
    )
    
    # 3. 加载模型
    print(f"Loading model: {config.model.name} from {args.model_path}")
    model = get_model(config.model.name)
    model.load(args.model_path)
    
    # 4. 评估循环（仅在测试集和拒绝集上评估）
    eval_splits = ['test', 'reject']  # 只评估这两个数据集
    
    for split_name in eval_splits:
        if split_name not in dataloaders:
            print(f"\nWarning: {split_name} set not found, skipping...")
            continue
            
        loader = dataloaders[split_name]
        print(f"\nEvaluating on {split_name} set...")
        
        # 收集真实标签和利润值
        y_true_list = []
        profit_list = []
        
        for _, y_batch, profit_batch in loader:
            y_true_list.append(y_batch.numpy())
            profit_list.append(profit_batch.numpy())
        
        y_true = np.concatenate(y_true_list)
        profits = np.concatenate(profit_list)
        
        # 获取模型预测
        y_pred = model.predict(loader)
        
        # 初始化指标字典
        metrics = {'Split': split_name}
        
        # 4.1 计算分类指标（仅二分类模型）
        if config.evaluation.model_type == 'classification':
            print(f"  Computing classification metrics...")
            cls_metrics = calculate_classification_metrics(y_true, y_pred)
            metrics.update(cls_metrics)
            
            # 打印分类指标
            print(f"    AUC: {cls_metrics['AUC']:.4f}")
            print(f"    Accuracy: {cls_metrics['Accuracy']:.4f}")
            print(f"    Precision: {cls_metrics['Precision']:.4f}")
            print(f"    Recall: {cls_metrics['Recall']:.4f}")
            print(f"    F1: {cls_metrics['F1']:.4f}")
        
        # 4.2 计算利润指标（所有模型）
        print(f"  Computing profit metrics...")
        profit_metrics = calculate_profit_metrics(
            y_pred, 
            profits, 
            model_type=config.evaluation.model_type,
            top_percent=config.evaluation.top_percent
        )
        metrics.update(profit_metrics)
        
        # 打印利润指标
        print(f"    Selected {profit_metrics['Selected_Count']} / {profit_metrics['Total_Samples']} samples ({config.evaluation.top_percent*100:.0f}%)")
        print(f"    Total Profit: {profit_metrics['Total_Profit']:.2f}")
        print(f"    Avg Profit: {profit_metrics['Avg_Profit']:.2f}")
    
    print("\n" + "="*50)
    print("✓ Evaluation completed.")
    print("="*50)


if __name__ == "__main__":
    main()

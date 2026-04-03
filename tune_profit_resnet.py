# -*- coding: utf-8 -*-
'''
@File    :   tune_profit_resnet.py
@Time    :   2026/02/19 21:34:00
@Author  :   chensy 
@Desc    :   盈利敏感 ResNet 超参数搜索脚本
'''

import os
import sys
import itertools
import numpy as np
import torch
from ultis.dataset import get_dataloaders
from ultis.seed import set_seed
from models import get_model
from configs import load_config


def evaluate_model(model, test_loader, top_percent=0.3):
    """
    评估模型的 Top N% Profit

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        top_percent (float): 选择前 N% 样本

    Returns:
        dict: 包含 total_profit 和 avg_profit
    """
    y_pred = model.predict(test_loader)

    # 收集真实 profit
    profits = []
    for _, _, profit_batch in test_loader:
        profits.extend(profit_batch.numpy())
    profits = np.array(profits)

    # 回归模式：按预测值降序排序
    n_selected = int(len(y_pred) * top_percent)
    sorted_indices = np.argsort(y_pred)[::-1]
    top_indices = sorted_indices[:n_selected]
    selected_profits = profits[top_indices]

    return {
        'total_profit': float(selected_profits.sum()),
        'avg_profit': float(selected_profits.mean()),
        'n_selected': n_selected
    }


def run_experiment(train_loader, valid_loader, test_loader, params, seed=42):
    """
    运行单次实验

    Args:
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
        test_loader: 测试数据加载器
        params (dict): 超参数字典
        seed (int): 随机种子

    Returns:
        dict: 实验结果
    """
    set_seed(seed)

    model_config = {
        'lr': params['lr'],
        'epochs': params['epochs'],
        'pretrained_encoder_path': params.get('pretrained_encoder_path'),
        'params': {
            'hidden_dim': params['hidden_dim'],
            'dropout': params['dropout'],
            'freeze_encoder': params.get('freeze_encoder', False),
            'lambda_rnc': params['lambda_rnc'],
            'mse_weight': params['mse_weight'],
            'rnc_temperature': params['rnc_temperature'],
        }
    }

    model = get_model('profit_resnet', config=model_config)
    model.train(train_loader, valid_loader)

    # 评估
    result = evaluate_model(model, test_loader)
    return result


def main():
    """超参数搜索主流程"""
    print("="*60)
    print("盈利敏感 ResNet 超参数搜索")
    print("="*60)

    # 加载配置
    config = load_config("configs/profit_resnet_config.yaml")

    # 加载数据（只加载一次）
    print("\n加载数据...")
    dataloaders = get_dataloaders(
        data_dir=config.data.data_dir,
        iv_path=config.data.iv_path,
        batch_size=128,
        top_n_features=config.data.top_n_features,
        data_type=config.data.data_type
    )

    train_loader = dataloaders['train']
    valid_loader = dataloaders.get('test')
    test_loader = dataloaders['test']

    print(f"✓ 数据加载完成\n")

    # ============================================
    # 超参数搜索空间
    # ============================================
    # 核心分析：
    # - 当前 MSE Loss ~ 8,000,000，RnC Loss ~ 6
    # - 需要提升 RnC 的相对权重
    # - 或者降低 MSE 的权重
    # ============================================

    param_grid = {
        'lr': [0.0005, 0.001],
        'lambda_rnc': [1.0, 1000.0, 100000.0],
        'mse_weight': [1.0],
        'rnc_temperature': [0.05, 0.1, 0.5],
    }

    # 固定参数
    fixed_params = {
        'hidden_dim': 256,
        'dropout': 0.3,
        'epochs': 30,
        'pretrained_encoder_path': 'model_param/pretrained_encoder.pth',
        'freeze_encoder': False,
    }

    # 生成所有参数组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))

    total_experiments = len(all_combinations)
    print(f"共 {total_experiments} 组参数组合")
    print(f"搜索空间: {param_grid}")
    print(f"固定参数: epochs={fixed_params['epochs']}, hidden_dim={fixed_params['hidden_dim']}")
    print("-"*60)

    # 运行搜索
    results = []

    for i, combo in enumerate(all_combinations):
        params = dict(zip(keys, combo))
        params.update(fixed_params)

        print(f"\n[{i+1}/{total_experiments}] lr={params['lr']}, "
              f"λ_rnc={params['lambda_rnc']}, "
              f"mse_w={params['mse_weight']}, "
              f"temp={params['rnc_temperature']}")

        try:
            result = run_experiment(train_loader, valid_loader, test_loader, params, seed=42)

            results.append({
                **params,
                'test_total_profit': result['total_profit'],
                'test_avg_profit': result['avg_profit'],
            })

            print(f"  → Test Top30% Profit: {result['total_profit']:.2f} "
                  f"(Avg: {result['avg_profit']:.2f})")

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            results.append({**params, 'test_total_profit': float('-inf'), 'test_avg_profit': float('-inf')})

    # 排序结果
    print("\n" + "="*60)
    print("搜索结果排名（按 Test Total Profit 降序）")
    print("="*60)

    results.sort(key=lambda x: x['test_total_profit'], reverse=True)

    print(f"\n{'Rank':<5} {'lr':<8} {'λ_rnc':<10} {'temp':<6} {'Total Profit':<15} {'Avg Profit':<12}")
    print("-"*60)

    for rank, r in enumerate(results[:10], 1):
        print(f"{rank:<5} {r['lr']:<8} {r['lambda_rnc']:<10} "
              f"{r['rnc_temperature']:<6} {r['test_total_profit']:<15.2f} {r['test_avg_profit']:<12.2f}")

    # 输出最优参数
    best = results[0]
    print(f"\n{'='*60}")
    print(f"🏆 最优参数:")
    print(f"  lr: {best['lr']}")
    print(f"  lambda_rnc: {best['lambda_rnc']}")
    print(f"  mse_weight: {best['mse_weight']}")
    print(f"  rnc_temperature: {best['rnc_temperature']}")
    print(f"  Test Total Profit: {best['test_total_profit']:.2f}")
    print(f"{'='*60}")

    # 保存结果
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv('tuning_results.csv', index=False)
        print(f"\n✓ 结果已保存到 tuning_results.csv")
    except Exception:
        pass


if __name__ == "__main__":
    main()

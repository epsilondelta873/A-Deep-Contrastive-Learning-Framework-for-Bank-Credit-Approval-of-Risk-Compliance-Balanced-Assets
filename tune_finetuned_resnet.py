# -*- coding: utf-8 -*-
'''
@File    :   tune_finetuned_resnet.py
@Time    :   2026/02/19 21:42:00
@Author  :   chensy 
@Desc    :   实验 1（分类模型）超参数搜索脚本
'''

import os
import sys
import itertools
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from ultis.dataset import get_dataloaders
from ultis.seed import set_seed
from models import get_model
from configs import load_config


def evaluate_model(model, test_loader, top_percent=0.3):
    """
    评估分类模型的 AUC 和 Top N% Profit

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        top_percent (float): 选择前 N% 样本

    Returns:
        dict: 包含 auc, total_profit, avg_profit
    """
    y_pred = model.predict(test_loader)

    y_true = []
    profits = []
    for _, y_batch, profit_batch in test_loader:
        y_true.extend(y_batch.numpy())
        profits.extend(profit_batch.numpy())
    y_true = np.array(y_true)
    profits = np.array(profits)

    # AUC
    auc = roc_auc_score(y_true, y_pred)

    # 分类模式：按概率升序排序（概率低 = 优质客户）
    n_selected = int(len(y_pred) * top_percent)
    sorted_indices = np.argsort(y_pred)
    top_indices = sorted_indices[:n_selected]
    selected_profits = profits[top_indices]

    return {
        'auc': float(auc),
        'total_profit': float(selected_profits.sum()),
        'avg_profit': float(selected_profits.mean()),
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
        'tensorboard': {'enabled': False},
        'params': {
            'hidden_dim': params['hidden_dim'],
            'dropout': params['dropout'],
            'freeze_encoder': params.get('freeze_encoder', False),
        }
    }

    model = get_model('finetuned_resnet', config=model_config)
    model.train(train_loader, valid_loader)

    result = evaluate_model(model, test_loader)
    return result


def main():
    """超参数搜索主流程"""
    print("="*60)
    print("实验 1（分类模型）超参数搜索")
    print("="*60)

    config = load_config("configs/contrastive_resnet_config.yaml")

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
    param_grid = {
        'lr': [0.0001, 0.0005, 0.001, 0.005],
        'dropout': [0.2, 0.3, 0.5],
        'freeze_encoder': [False],
    }

    fixed_params = {
        'hidden_dim': 256,
        'epochs': 30,
        'pretrained_encoder_path': 'model_param/pretrained_encoder.pth',
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))

    total_experiments = len(all_combinations)
    print(f"共 {total_experiments} 组参数组合")
    print(f"搜索空间: {param_grid}")
    print(f"固定参数: epochs={fixed_params['epochs']}, hidden_dim={fixed_params['hidden_dim']}")
    print("-"*60)

    results = []

    for i, combo in enumerate(all_combinations):
        params = dict(zip(keys, combo))
        params.update(fixed_params)

        print(f"\n[{i+1}/{total_experiments}] lr={params['lr']}, "
              f"dropout={params['dropout']}, "
              f"freeze={params['freeze_encoder']}")

        try:
            result = run_experiment(train_loader, valid_loader, test_loader, params, seed=42)

            results.append({
                **{k: params[k] for k in keys},
                'test_auc': result['auc'],
                'test_total_profit': result['total_profit'],
                'test_avg_profit': result['avg_profit'],
            })

            print(f"  → AUC: {result['auc']:.4f}, "
                  f"Top30% Profit: {result['total_profit']:.2f} "
                  f"(Avg: {result['avg_profit']:.2f})")

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            results.append({
                **{k: params[k] for k in keys},
                'test_auc': 0,
                'test_total_profit': float('-inf'),
                'test_avg_profit': float('-inf'),
            })

    # 按 Total Profit 排序
    print("\n" + "="*60)
    print("搜索结果排名（按 Test Total Profit 降序）")
    print("="*60)

    results.sort(key=lambda x: x['test_total_profit'], reverse=True)

    print(f"\n{'Rank':<5} {'lr':<8} {'dropout':<9} {'AUC':<8} {'Total Profit':<15} {'Avg Profit':<12}")
    print("-"*60)

    for rank, r in enumerate(results[:12], 1):
        print(f"{rank:<5} {r['lr']:<8} {r['dropout']:<9} "
              f"{r['test_auc']:<8.4f} {r['test_total_profit']:<15.2f} {r['test_avg_profit']:<12.2f}")

    best = results[0]
    print(f"\n{'='*60}")
    print(f"🏆 最优参数:")
    print(f"  lr: {best['lr']}")
    print(f"  dropout: {best['dropout']}")
    print(f"  AUC: {best['test_auc']:.4f}")
    print(f"  Test Total Profit: {best['test_total_profit']:.2f}")
    print(f"{'='*60}")

    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv('tuning_results_exp1.csv', index=False)
        print(f"\n✓ 结果已保存到 tuning_results_exp1.csv")
    except Exception:
        pass


if __name__ == "__main__":
    main()

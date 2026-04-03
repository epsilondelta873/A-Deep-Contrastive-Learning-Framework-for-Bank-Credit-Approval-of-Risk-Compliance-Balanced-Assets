# -*- coding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2026/01/17 20:53:00
@Author  :   chensy 
@Desc    :   通用模型训练脚本
'''

import argparse
import os
import torch
from ultis.dataset import get_dataloaders
from ultis.seed import set_seed
from models import get_model
from configs import load_config, merge_args_with_config, Config


def main():
    """
    主训练流程
    
    该函数执行以下步骤：
    1. 解析命令行参数
    2. 加载训练数据
    3. 初始化模型
    4. 执行训练
    5. 保存模型
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="通用模型训练脚本")
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
        "--lr",
        type=float,
        default=None,
        help="学习率，会覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数，会覆盖配置文件中的设置"
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
        "--output_dir", 
        type=str, 
        default=None,
        help="模型保存目录，会覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="训练批次大小，会覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--top_n_features", 
        type=int, 
        default=None,
        help="选择 IV 值最高的前 N 个特征，会覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于保证实验可复现性（默认 42）"
    )
    
    args = parser.parse_args()
    
    # 步骤 0: 加载配置文件并合并命令行参数
    print(f"Loading configuration from {args.config}...")
    try:
        config = load_config(args.config)
        config = merge_args_with_config(config, args)
    except FileNotFoundError:
        print(f"Warning: Config file {args.config} not found, using command-line arguments only.")
        # 如果配置文件不存在，使用默认配置
        config = Config({
            'model': {'name': args.model_name or 'baseline', 'lr': args.lr or 0.01, 'epochs': args.epochs or 20},
            'data': {
                'data_dir': args.data_dir, 
                'iv_path': args.iv_path,
                'batch_size': args.batch_size or 64,
                'top_n_features': args.top_n_features or 20
            },
            'training': {'output_dir': args.output_dir or 'checkpoints'}
        })
    
    # 打印当前使用的配置
    print("\n" + "="*50)
    print("当前配置:")
    print(f"  模型: {config.model.name}")
    print(f"  学习率: {config.model.lr}")
    print(f"  训练轮数: {config.model.epochs}")
    print(f"  批次大小: {config.data.batch_size}")
    print(f"  特征数量: {config.data.top_n_features}")
    print(f"  数据类型: {config.data.data_type}")  # 新增：显示数据类型
    print(f"  输出目录: {config.training.output_dir}")
    print("="*50 + "\n")
    
    # 步骤 0.5: 设置随机种子以保证实验可复现性
    set_seed(args.seed)
    print()  # 空行分隔
    
    # 步骤 1: 加载数据
    print(f"Loading data...")
    dataloaders = get_dataloaders(
        data_dir=config.data.data_dir,
        iv_path=config.data.iv_path,
        batch_size=config.data.batch_size,
        top_n_features=config.data.top_n_features,
        data_type=config.data.data_type  # 使用配置文件中的数据类型
    )
    
    if 'train' not in dataloaders:
        raise ValueError("训练数据集未找到！")
        
    train_loader = dataloaders['train']
    valid_loader = dataloaders.get('test')  # 使用测试集作为验证集（可选）
    
    # 步骤 2: 初始化模型
    print(f"Initializing model: {config.model.name}")
    # 将模型配置传递给模型（包括通用参数和模型特定参数）
    model_config = {
        'lr': config.model.lr,
        'epochs': config.model.epochs,
        'seed': args.seed,  # 传递用于树模型的全局种子
        'evaluation': config.get('evaluation', {}),
        'params': config.model.get('params', {})  # 模型特定参数
    }
    model = get_model(config.model.name, config=model_config)
    
    # 步骤 3: 训练模型
    print("Training started...")
    model.train(train_loader, valid_loader)
    print("Training finished.")
    
    # 步骤 4: 保存模型
    output_dir = config.training.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, f"{config.model.name}_model.pkl")
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()

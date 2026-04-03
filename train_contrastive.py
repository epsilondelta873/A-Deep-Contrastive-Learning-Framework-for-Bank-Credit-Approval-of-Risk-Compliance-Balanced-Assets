# -*- coding: utf-8 -*-
'''
@File    :   train_contrastive.py
@Time    :   2026/02/08 19:30:00
@Author  :   chensy 
@Desc    :   对比学习预训练脚本
'''

import argparse
import os
import sys
from ultis.contrastive_dataset import get_contrastive_dataloader
from ultis.dataset import select_features
from ultis.seed import set_seed
from models import get_model
from configs import load_config, merge_args_with_config, Config


def main():
    """
    对比学习预训练主流程
    
    步骤：
    1. 解析命令行参数
    2. 加载配置文件
    3. 创建对比学习数据集（混合 train_accepted + train_rejected）
    4. 初始化对比学习模型
    5. 执行预训练
    6. 保存编码器权重
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="对比学习预训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/contrastive_resnet_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="数据目录路径"
    )
    parser.add_argument(
        "--iv_path",
        type=str,
        default=None,
        help="IV 结果文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="编码器保存目录"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="实验名称"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    # 步骤 1: 加载配置文件
    print(f"Loading configuration from {args.config}...")
    try:
        config = load_config(args.config)
        config = merge_args_with_config(config, args)
    except FileNotFoundError:
        print(f"Error: Config file {args.config} not found!")
        sys.exit(1)
    
    # 打印配置
    print("\n" + "="*50)
    print("对比学习预训练配置:")
    print(f"  数据目录: {config.data.data_dir}")
    print(f"  数据类型: {config.data.data_type}")
    print(f"  批次大小: {config.data.batch_size}")
    print(f"  学习率: {config.model.lr}")
    print(f"  训练轮数: {config.model.epochs}")
    print(f"  温度参数: {config.model.params.temperature}")
    print(f"  输出目录: {config.training.output_dir}")
    print("="*50 + "\n")
    
    # 步骤 2: 设置随机种子
    set_seed(args.seed)
    print()
    
    # 步骤 3: 准备数据
    print("准备对比学习数据...")
    
    # 3.1 选择特征
    data_dir = config.data.data_dir
    iv_path = config.data.iv_path
    top_n_features = config.data.top_n_features
    data_type = config.data.data_type
    
    feature_names = None
    if top_n_features is not None:
        feature_names = select_features(iv_path, top_n=top_n_features, data_type=data_type)
        print(f"Selected {len(feature_names)} features")
    else:
        print("Using all features")
    
    # 3.2 创建对比学习数据加载器
    # 混合 train_accepted 和 train_rejected
    file_names = [
        f"train_accepted_{data_type}.xlsx",
        f"train_rejected_{data_type}.xlsx"
    ]
    
    # 数据增强配置
    augmentation_config = {
        'noise_level': config.training.augmentation.noise_level,
        'drop_prob': config.training.augmentation.drop_prob,
        'use_both': config.training.augmentation.get('use_both', True)
    }
    
    contrastive_loader = get_contrastive_dataloader(
        data_dir=data_dir,
        file_names=file_names,
        feature_names=feature_names,
        batch_size=config.data.batch_size,
        augmentation_config=augmentation_config
    )
    
    print(f"✓ 对比学习数据加载器创建成功")
    print(f"  总 batch 数: {len(contrastive_loader)}")
    print(f"  批次大小: {config.data.batch_size}\n")
    
    # 步骤 4: 初始化对比学习模型
    print("初始化对比学习模型...")
    model_config = {
        'lr': config.model.lr,
        'epochs': config.model.epochs,
        'experiment_name': args.experiment_name,
        'tensorboard': config.training.get('tensorboard', {}),
        'params': config.model.get('params', {})
    }
    
    model = get_model('contrastive_resnet', config=model_config)
    
    # 步骤 5: 开始预训练
    print("\n" + "="*50)
    print("开始对比学习预训练...")
    print("="*50)
    model.train(contrastive_loader)
    
    # 步骤 6: 保存编码器
    output_dir = config.training.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    encoder_path = os.path.join(output_dir, "pretrained_encoder.pth")
    model.save_encoder(encoder_path)
    
    print(f"\n" + "="*50)
    print("对比学习预训练完成！")
    print(f"编码器已保存到: {encoder_path}")
    print("\n下一步: 使用 train_finetune.py 进行有监督微调")
    print("="*50)


if __name__ == "__main__":
    main()

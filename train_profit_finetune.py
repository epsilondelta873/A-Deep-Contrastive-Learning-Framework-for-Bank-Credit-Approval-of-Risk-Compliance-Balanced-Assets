# -*- coding: utf-8 -*-
'''
@File    :   train_profit_finetune.py
@Time    :   2026/02/19 21:25:00
@Author  :   chensy 
@Desc    :   盈利敏感有监督微调脚本（实验 2）
'''

import argparse
import os
import sys
from ultis.dataset import get_dataloaders
from ultis.seed import set_seed
from models import get_model
from configs import load_config, merge_args_with_config, Config


def main():
    """
    盈利敏感有监督微调主流程
    
    步骤：
    1. 解析命令行参数
    2. 加载配置文件
    3. 加载有标签训练数据（仅 train_accepted，含 profit 标签）
    4. 初始化盈利敏感模型并加载预训练编码器
    5. 执行 MSE + Rank-N-Contrast 双损失微调
    6. 保存最终模型
    """
    parser = argparse.ArgumentParser(description="盈利敏感有监督微调脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/profit_resnet_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--pretrained_encoder",
        type=str,
        default=None,
        help="预训练编码器路径"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="数据目录路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="模型保存目录"
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
    
    # 确定预训练编码器路径
    if args.pretrained_encoder:
        pretrained_encoder_path = args.pretrained_encoder
    else:
        output_dir = config.training.output_dir if args.output_dir is None else args.output_dir
        pretrained_encoder_path = os.path.join(output_dir, "pretrained_encoder.pth")
    
    use_pretrained = os.path.exists(pretrained_encoder_path)
    
    # 打印配置
    print("\n" + "="*50)
    print("盈利敏感微调配置:")
    print(f"  数据目录: {config.data.data_dir}")
    print(f"  数据类型: {config.data.data_type}")
    print(f"  批次大小: {config.finetune.batch_size}")
    print(f"  学习率: {config.finetune.lr}")
    print(f"  训练轮数: {config.finetune.epochs}")
    print(f"  λ_RnC: {config.finetune.lambda_rnc}")
    print(f"  MSE权重: {config.finetune.mse_weight}")
    print(f"  RnC温度: {config.finetune.rnc_temperature}")
    print(f"  Weight Decay: {config.finetune.weight_decay}")
    print(f"  编码器LR缩放: {config.finetune.encoder_lr_scale}")
    print(f"  回归头LR缩放: {config.finetune.head_lr_scale}")
    print(f"  梯度裁剪: {config.finetune.grad_clip_norm}")
    print(f"  验证Top百分比: {config.evaluation.top_percent}")
    
    if use_pretrained:
        print(f"  预训练编码器: {pretrained_encoder_path} ✓")
        print(f"  冻结编码器: {config.finetune.freeze_layers > 0}")
    else:
        print(f"  预训练编码器: 无（将使用随机初始化）⚠")
    
    print(f"  输出目录: {config.training.output_dir}")
    print("="*50 + "\n")
    
    # 步骤 2: 设置随机种子
    set_seed(args.seed)
    print()
    
    # 步骤 3: 加载数据
    print("加载有标签训练数据...")
    
    dataloaders = get_dataloaders(
        data_dir=config.data.data_dir,
        iv_path=config.data.iv_path,
        batch_size=config.finetune.batch_size,
        top_n_features=config.data.top_n_features,
        data_type=config.data.data_type
    )
    
    if 'train' not in dataloaders:
        print("Error: 训练数据集未找到！")
        sys.exit(1)
    
    train_loader = dataloaders['train']
    valid_loader = dataloaders.get('test')
    
    print(f"✓ 数据加载完成")
    print(f"  训练集 batch 数: {len(train_loader)}")
    if valid_loader:
        print(f"  验证集 batch 数: {len(valid_loader)}\n")
    
    # 步骤 4: 初始化模型
    print("初始化盈利敏感模型...")
    
    model_config = {
        'lr': config.finetune.lr,
        'epochs': config.finetune.epochs,
        'pretrained_encoder_path': pretrained_encoder_path if use_pretrained else None,
        'params': {
            'hidden_dim': config.model.params.get('hidden_dim', 128),
            'dropout': config.model.params.get('dropout', 0.3),
            'freeze_encoder': config.finetune.freeze_layers > 0,
            'lambda_rnc': config.finetune.lambda_rnc,
            'mse_weight': config.finetune.mse_weight,
            'rnc_temperature': config.finetune.rnc_temperature,
            'weight_decay': config.finetune.weight_decay,
            'encoder_lr_scale': config.finetune.encoder_lr_scale,
            'head_lr_scale': config.finetune.head_lr_scale,
            'grad_clip_norm': config.finetune.grad_clip_norm,
            'top_percent': config.evaluation.top_percent,
        }
    }
    
    model = get_model('profit_resnet', config=model_config)
    
    # 步骤 5: 开始微调
    print("\n" + "="*50)
    print("开始盈利敏感有监督微调...")
    print("="*50)
    model.train(train_loader, valid_loader)
    
    # 步骤 6: 保存模型
    output_dir = config.training.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = os.path.join(output_dir, "profit_resnet_model.pkl")
    model.save(model_path)
    
    print(f"\n" + "="*50)
    print("盈利敏感微调完成！")
    print(f"模型已保存到: {model_path}")
    print("\n下一步: 使用 evaluate.py 评估模型性能")
    print(f"  python evaluate.py --model_path {model_path} --model_name profit_resnet --model_type regression --config {args.config}")
    print("="*50)


if __name__ == "__main__":
    main()

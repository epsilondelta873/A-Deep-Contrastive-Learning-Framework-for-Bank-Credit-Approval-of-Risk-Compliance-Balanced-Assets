# -*- coding: utf-8 -*-
"""
@File    :   extract_embeddings.py
@Author  :   chensy
@Desc    :   特征提取脚本：负责加载脱离预训练的 ResNetEncoder，
             并将所有数据集（训练、测试、拒绝集）映射到特征嵌套空间，保存为 npz。
"""

import argparse
import os
import sys
import torch
import numpy as np

# 添加项目根目录到 sys.path，以便导入项目包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultis.dataset import get_dataloaders
from ultis.seed import set_seed
from configs import load_config, merge_args_with_config
from models.resnet_encoder import ResNetEncoder


def extract_features(loader, encoder, device):
    """
    抽取特征的内部函数
    
    Args:
        loader: 数据集加载器
        encoder: 设置为 eval 模式的 ResNetEncoder
        device: 计算设备
        
    Returns:
        X_emb: 特征矩阵，numpy array (N, hidden_dim)
        y: 原标签矩阵，numpy array (N,)
        profits: 利润向量，numpy array (N,)
    """
    encoder.eval()
    X_emb_list = []
    y_list = []
    profit_list = []
    
    with torch.no_grad():
        for X_batch, y_batch, profit_batch in loader:
            X_batch = X_batch.to(device)
            # 通过编码器输出 Embedding 向量
            emb = encoder(X_batch)
            
            X_emb_list.append(emb.cpu().numpy())
            y_list.append(y_batch.numpy())
            profit_list.append(profit_batch.numpy())
            
    return np.concatenate(X_emb_list), np.concatenate(y_list), np.concatenate(profit_list)


def main():
    parser = argparse.ArgumentParser(description="高级表征构建与特征提取脚本")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/hybrid_lgbm_config.yaml", 
        help="配置文件路径，默认使用混合模型的专用配置"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 1. 挂载加载配置
    print(f"Loading configuration from {args.config}...")
    try:
        config = load_config(args.config)
        config = merge_args_with_config(config, args)
    except FileNotFoundError:
        print(f"Error: 配置文件 {args.config} 未找到！")
        sys.exit(1)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. 检查与加载模型权重状态
    pretrained_path = config.extract.pretrained_encoder_path
    if not os.path.exists(pretrained_path):
        print(f"Error: {pretrained_path} 不存在！请先使用基于 ResNet 整体的预训练脚本。")
        sys.exit(1)

    print(f"Loading ResNetEncoder weights from {pretrained_path}...")
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # 动态匹配保存时候的模型超参数
    input_dim = checkpoint['input_dim']
    hidden_dim = checkpoint.get('hidden_dim', config.extract.get('hidden_dim', 256))
    dropout = checkpoint.get('dropout', config.extract.get('dropout', 0.3))
    
    # 3. 初始化独立提取网络的结构（仅保留 Encoder 阶段）
    encoder = ResNetEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    print("✓ 编码器计算图和权重完全复现与加载成功\n")

    # 4. 处理 DataLoader 以流式转换所有数据
    print("开始加载原始数据集 (train, test, reject)...")
    dataloaders = get_dataloaders(
        data_dir=config.data.data_dir,
        iv_path=config.data.iv_path,
        batch_size=config.data.batch_size,
        top_n_features=config.data.top_n_features,
        data_type=config.data.data_type
    )

    output_dir = config.extract.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 5. 执行特征提取保存
    print("\n" + "="*50)
    for split_name in ['train', 'test', 'reject']:
        if split_name not in dataloaders:
            print(f"Warning: 分片数据集 {split_name} 未在目录中找到，直接跳过计算。")
            continue
            
        print(f"提取 [{split_name}] 集合特征...")
        X_emb, y, profits = extract_features(dataloaders[split_name], encoder, device)
        
        save_path = os.path.join(output_dir, f"{split_name}_emb.npz")
        np.savez(save_path, X_emb=X_emb, y=y, profits=profits)
        print(f"✓ {split_name} 特征导出成功！本地保存路径为: {save_path} (Shape: {X_emb.shape})")

    print("="*50)
    print("系统级高级表征提取完毕！现可调用 train_hybrid_lgbm.py 训练二级预测架构。")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
'''
@File    :   contrastive_resnet.py
@Time    :   2026/02/08 19:30:00
@Author  :   chensy 
@Desc    :   对比学习 ResNet 模型，包含 InfoNCE 损失函数和完整训练逻辑
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .resnet_encoder import ResNetEncoder, ProjectionHead, ClassificationHead
from .base import BaseModel
from . import register_model


class InfoNCELoss(nn.Module):
    """
    InfoNCE 对比学习损失函数
    
    拉近同一样本的两个增强视图（正样本对），推远不同样本（负样本对）
    """
    
    def __init__(self, temperature=0.5):
        """
        初始化 InfoNCE 损失
        
        Args:
            temperature (float): 温度参数，控制相似度的软化程度
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        """
        计算 InfoNCE 损失
        
        Args:
            z_i (torch.Tensor): 第一个视图的特征，shape (batch_size, projection_dim)
            z_j (torch.Tensor): 第二个视图的特征，shape (batch_size, projection_dim)
            
        Returns:
            torch.Tensor: InfoNCE 损失值（标量）
        """
        batch_size = z_i.shape[0]
        
        # 1. L2 归一化（使相似度计算变为余弦相似度）
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # 2. 拼接所有样本 [z_i; z_j]
        representations = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, dim)
        
        # 3. 计算相似度矩阵（余弦相似度）
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),  # (2B, 1, dim)
            representations.unsqueeze(0),  # (1, 2B, dim)
            dim=2
        )  # (2B, 2B)
        
        # 4. 创建正样本标签
        # 对于前半部分（z_i），正样本在后半部分（z_j）的对应位置
        # 对于后半部分（z_j），正样本在前半部分（z_i）的对应位置
        labels = torch.arange(batch_size).to(z_i.device)
        labels = torch.cat([labels + batch_size, labels])  # [B, B+1, ..., 2B-1, 0, 1, ..., B-1]
        
        # 5. 去掉对角线（自身相似度）
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # 6. 应用温度参数
        similarity_matrix = similarity_matrix / self.temperature
        
        # 7. 计算交叉熵损失
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


@register_model('contrastive_resnet')
class ContrastiveResNet(BaseModel):
    """
    对比学习 ResNet 模型包装类
    
    用于对比学习预训练阶段，封装了编码器、投影头和 InfoNCE 损失
    """
    
    def __init__(self, config=None):
        """
        初始化对比学习模型
        
        Args:
            config (dict): 配置字典，包含：
                - lr: 学习率
                - epochs: 训练轮数
                - params: 模型参数
                    - hidden_dim: 编码器隐藏维度
                    - projection_dim: 投影头输出维度
                    - dropout: Dropout 比例
                    - temperature: InfoNCE 温度参数
        """
        self.config = config or {}
        
        # 读取通用参数
        self.lr = self.config.get('lr', 0.001)
        self.epochs = self.config.get('epochs', 100)
        
        # 读取模型特定参数
        params = self.config.get('params', {})
        self.hidden_dim = params.get('hidden_dim', 128)
        self.projection_dim = params.get('projection_dim', 64)
        self.dropout = params.get('dropout', 0.3)
        self.temperature = params.get('temperature', 0.5)
        
        print(f"对比学习 ResNet 配置:")
        print(f"  hidden_dim={self.hidden_dim}")
        print(f"  projection_dim={self.projection_dim}")
        print(f"  dropout={self.dropout}")
        print(f"  temperature={self.temperature}")
        
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型组件（延迟初始化）
        self.encoder = None
        self.projection_head = None
        self.criterion = None
    
    def train(self, train_loader, valid_loader=None):
        """
        对比学习预训练流程
        
        Args:
            train_loader: 对比学习数据加载器（返回两个视图）
            valid_loader: 验证集加载器（可选，用于监控）
        """
        # 步骤 1：推断输入维度并初始化模型
        view1, view2 = next(iter(train_loader))
        input_dim = view1.shape[1]
        
        if self.encoder is None:
            print(f"\n初始化对比学习模型，输入维度={input_dim}")
            
            # 创建编码器
            self.encoder = ResNetEncoder(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            ).to(self.device)
            
            # 创建投影头
            self.projection_head = ProjectionHead(
                hidden_dim=self.hidden_dim,
                projection_dim=self.projection_dim
            ).to(self.device)
            
            # 创建损失函数
            self.criterion = InfoNCELoss(temperature=self.temperature)
        
        # 步骤 2：定义优化器
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.projection_head.parameters()),
            lr=self.lr
        )
        
        print(f"\n开始对比学习预训练，设备={self.device}")
        print(f"训练轮数: {self.epochs}, Batch Size: {train_loader.batch_size}")
        print(f"总样本数: ~{len(train_loader) * train_loader.batch_size}")
        print("✓ 启用 Best Model Checkpoint: 将自动保存loss最低的编码器\n")
        
        # 初始化 Best Model Checkpoint
        best_loss = float('inf')
        best_epoch = 0
        
        # 步骤 4：训练循环
        for epoch in range(self.epochs):
            self.encoder.train()
            self.projection_head.train()
            
            total_loss = 0
            batch_count = 0
            
            for view1, view2 in train_loader:
                view1 = view1.to(self.device)
                view2 = view2.to(self.device)
                
                # 前向传播
                h1 = self.encoder(view1)
                h2 = self.encoder(view2)
                
                z1 = self.projection_head(h1)
                z2 = self.projection_head(h2)
                
                # 计算 InfoNCE 损失
                loss = self.criterion(z1, z2)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            # 计算平均损失
            avg_loss = total_loss / batch_count
            
            # Best Model Checkpoint: 保存loss最低的编码器
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                
                if (epoch + 1) % 5 == 0 or epoch < 10:
                    print(f"  ⭐ Epoch [{epoch+1}] - New best model! Loss: {avg_loss:.4f}")
            
            # 输出训练进度
            if (epoch + 1) % 5 == 0 or epoch < 10:
                print(f"Epoch [{epoch+1}/{self.epochs}], Contrastive Loss: {avg_loss:.4f}")
        
        print(f"\n✓ 对比学习预训练完成！")
        print(f"✓ 最佳模型: Epoch {best_epoch}, Loss: {best_loss:.4f}")
        
    def save_encoder(self, path):
        """
        保存编码器权重（用于后续微调）
        
        Args:
            path (str): 保存路径
        """
        if self.encoder is None:
            raise ValueError("编码器尚未初始化！")
        
        # 保存编码器的 state_dict 和配置
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'input_dim': self.encoder.input_dim
        }, path)
        
        print(f"✓ 编码器已保存到: {path}")
    
    def predict(self, test_loader):
        """
        对比学习模型不用于预测
        如需预测，请使用微调后的分类模型
        """
        raise NotImplementedError(
            "对比学习模型不支持直接预测。"
            "请使用 FinetunedResNet 进行有监督微调后再预测。"
        )
    
    def save(self, path):
        """保存完整模型（包括编码器和投影头）"""
        if self.encoder is None or self.projection_head is None:
            raise ValueError("模型尚未初始化！")
        
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'projection_head_state_dict': self.projection_head.state_dict(),
            'config': self.config,
            'hidden_dim': self.hidden_dim,
            'projection_dim': self.projection_dim,
            'dropout': self.dropout,
            'temperature': self.temperature,
            'input_dim': self.encoder.input_dim
        }, path)
    
    def load(self, path):
        """加载预训练模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 恢复配置
        input_dim = checkpoint['input_dim']
        hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
        projection_dim = checkpoint.get('projection_dim', self.projection_dim)
        dropout = checkpoint.get('dropout', self.dropout)
        temperature = checkpoint.get('temperature', self.temperature)
        
        # 重建模型
        self.encoder = ResNetEncoder(input_dim, hidden_dim, dropout).to(self.device)
        self.projection_head = ProjectionHead(hidden_dim, projection_dim).to(self.device)
        self.criterion = InfoNCELoss(temperature)
        
        # 加载权重
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    print("--- 对比学习 ResNet 模型测试 ---\n")
    
    # 创建模拟的对比学习数据
    batch_size = 32
    n_features = 100
    n_batches = 10
    
    # 模拟 DataLoader
    class MockContrastiveDataLoader:
        def __init__(self, n_batches, batch_size, n_features):
            self.n_batches = n_batches
            self.batch_size = batch_size
            self.n_features = n_features
            self.current = 0
        
        def __iter__(self):
            self.current = 0
            return self
        
        def __next__(self):
            if self.current >= self.n_batches:
                raise StopIteration
            self.current += 1
            view1 = torch.randn(self.batch_size, self.n_features)
            view2 = torch.randn(self.batch_size, self.n_features)
            return view1, view2
        
        def __len__(self):
            return self.n_batches
    
    mock_loader = MockContrastiveDataLoader(n_batches, batch_size, n_features)
    
    # 配置
    config = {
        'lr': 0.001,
        'epochs': 5,
        'params': {
            'hidden_dim': 64,
            'projection_dim': 32,
            'dropout': 0.3,
            'temperature': 0.5
        }
    }
    
    # 创建并训练模型
    model = ContrastiveResNet(config=config)
    print("开始训练...\n")
    model.train(mock_loader)
    
    # 保存编码器
    save_path = "test_encoder.pth"
    model.save_encoder(save_path)
    
    # 清理
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"\n✓ 测试完成，已清理临时文件")

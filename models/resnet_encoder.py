# -*- coding: utf-8 -*-
'''
@File    :   resnet_encoder.py
@Time    :   2026/02/08 19:30:00
@Author  :   chensy 
@Desc    :   ResNet 编码器模块（对比学习版本），包含编码器、投影头和分类头
'''

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    残差块（Residual Block）
    
    实现 y = F(x) + x 的残差学习
    """
    
    def __init__(self, hidden_dim, dropout=0.3):
        """
        初始化残差块
        
        Args:
            hidden_dim (int): 隐藏层维度
            dropout (float): Dropout 比例
        """
        super(ResidualBlock, self).__init__()
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.relu_final = nn.ReLU()
    
    def forward(self, x):
        """前向传播"""
        identity = x
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.relu_final(out)
        
        return out


class ResNetEncoder(nn.Module):
    """
    ResNet 编码器（不含分类头）
    
    用于对比学习预训练，输出特征向量而非分类结果
    """
    
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        """
        初始化编码器
        
        Args:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度（编码器输出维度）
            dropout (float): Dropout 比例
        """
        super(ResNetEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 投影层：将输入映射到隐藏维度
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 残差块
        self.residual_block = ResidualBlock(hidden_dim, dropout)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征，shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: 特征向量，shape (batch_size, hidden_dim)
        """
        x = self.projection(x)
        x = self.residual_block(x)
        return x


class ProjectionHead(nn.Module):
    """
    投影头（Projection Head）
    
    用于对比学习，将编码器输出映射到对比学习空间
    预训练完成后会被丢弃
    """
    
    def __init__(self, hidden_dim=128, projection_dim=64):
        """
        初始化投影头
        
        Args:
            hidden_dim (int): 输入维度（编码器输出维度）
            projection_dim (int): 投影空间维度
        """
        super(ProjectionHead, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 编码器输出，shape (batch_size, hidden_dim)
            
        Returns:
            torch.Tensor: 投影向量，shape (batch_size, projection_dim)
        """
        return self.net(x)


class ClassificationHead(nn.Module):
    """
    分类头（Classification Head）
    
    用于有监督微调，将编码器输出映射到分类概率
    """
    
    def __init__(self, hidden_dim=128):
        """
        初始化分类头
        
        Args:
            hidden_dim (int): 输入维度（编码器输出维度）
        """
        super(ClassificationHead, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 编码器输出，shape (batch_size, hidden_dim)
            
        Returns:
            torch.Tensor: 分类概率，shape (batch_size, 1)
        """
        return self.net(x)


class RegressionHead(nn.Module):
    """
    回归头（Regression Head）
    
    用于盈利敏感微调，将编码器输出映射到连续盈利预测值
    """
    
    def __init__(self, hidden_dim=128):
        """
        初始化回归头
        
        Args:
            hidden_dim (int): 输入维度（编码器输出维度）
        """
        super(RegressionHead, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 编码器输出，shape (batch_size, hidden_dim)
            
        Returns:
            torch.Tensor: 盈利预测值，shape (batch_size, 1)
        """
        return self.net(x)


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    print("--- ResNet 编码器模块测试 ---\n")
    
    # 配置
    batch_size = 16
    input_dim = 100
    hidden_dim = 128
    projection_dim = 64
    
    # 创建示例数据
    x = torch.randn(batch_size, input_dim)
    
    # 1. 测试编码器
    print("1. 测试 ResNetEncoder")
    encoder = ResNetEncoder(input_dim, hidden_dim)
    h = encoder(x)
    print(f"   输入 shape: {x.shape}")
    print(f"   编码器输出 shape: {h.shape}")
    assert h.shape == (batch_size, hidden_dim), "编码器输出维度错误！"
    print("   ✓ 编码器测试通过\n")
    
    # 2. 测试投影头
    print("2. 测试 ProjectionHead")
    projection_head = ProjectionHead(hidden_dim, projection_dim)
    z = projection_head(h)
    print(f"   编码器输出 shape: {h.shape}")
    print(f"   投影头输出 shape: {z.shape}")
    assert z.shape == (batch_size, projection_dim), "投影头输出维度错误！"
    print("   ✓ 投影头测试通过\n")
    
    # 3. 测试分类头
    print("3. 测试 ClassificationHead")
    classification_head = ClassificationHead(hidden_dim)
    y_pred = classification_head(h)
    print(f"   编码器输出 shape: {h.shape}")
    print(f"   分类头输出 shape: {y_pred.shape}")
    print(f"   输出值范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    assert y_pred.shape == (batch_size, 1), "分类头输出维度错误！"
    assert (y_pred >= 0).all() and (y_pred <= 1).all(), "分类头输出不在 [0, 1] 范围内！"
    print("   ✓ 分类头测试通过\n")
    
    # 4. 测试完整流程
    print("4. 测试完整流程（编码器 → 投影头）")
    full_model = nn.Sequential(encoder, projection_head)
    z_full = full_model(x)
    print(f"   输入 shape: {x.shape}")
    print(f"   最终输出 shape: {z_full.shape}")
    assert z_full.shape == (batch_size, projection_dim), "完整流程输出维度错误！"
    print("   ✓ 完整流程测试通过\n")
    
    print("="*50)
    print("所有测试通过！编码器模块工作正常。")

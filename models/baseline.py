# -*- coding: utf-8 -*-
'''
@File    :   baseline.py
@Time    :   2026/01/17 20:37:46
@Author  :   chensy 
@Desc    :   Baseline 模型实现：基于 PyTorch 的逻辑回归模型
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from . import register_model
from .base import BaseModel


class LogisticRegression(nn.Module):
    """
    逻辑回归神经网络模型
    
    该类实现了一个单层线性网络，用于二分类任务。
    网络结构：Linear(input_dim, 1) -> Sigmoid
    """
    def __init__(self, input_dim):
        """
        初始化网络层
        
        Args:
            input_dim: 输入特征的维度
        """
        super(LogisticRegression, self).__init__()
        # 定义线性层: y = wx + b
        # 输入维度为 input_dim，输出维度为 1（二分类概率）
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，shape 为 (batch_size, input_dim)
            
        Returns:
            输出概率，shape 为 (batch_size, 1)，值域为 [0, 1]
        """
        return torch.sigmoid(self.linear(x))


@register_model('baseline')
class BaselineCheck(BaseModel):
    """
    基线模型包装类
    
    该类封装了逻辑回归模型的训练、预测、保存和加载逻辑，
    继承自 BaseModel 以保证接口统一性。
    """
    def __init__(self, config=None):
        """
        初始化模型配置
        
        Args:
            config: 配置字典，可包含以下参数：
                - lr: 学习率，默认 0.01
                - epochs: 训练轮数，默认 20
                - experiment_name: 实验名称（可选）
                - tensorboard: TensorBoard 配置字典（可选）
        """
        self.config = config or {}
        
        # 读取超参数配置
        self.lr = self.config.get('lr', 0.01)
        self.epochs = self.config.get('epochs', 20)
        
        # 读取 TensorBoard 配置
        tensorboard_config = self.config.get('tensorboard', {})
        self.tensorboard_enabled = tensorboard_config.get('enabled', True)
        self.tensorboard_log_dir = tensorboard_config.get('log_dir', 'runs')
        self.experiment_name = self.config.get('experiment_name', None)
        
        # 设备选择：优先使用 GPU，若不可用则使用 CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型实例，延迟初始化（需要先获取输入维度）
        self.model = None

    def train(self, train_loader, valid_loader=None):
        """
        模型训练流程
        
        Args:
            train_loader: 训练数据的 DataLoader
            valid_loader: 验证数据的 DataLoader（可选）
        """
        # 步骤 1：推断输入维度并初始化模型
        first_batch_X, _, _ = next(iter(train_loader))  # 解包三元组，忽略 y 和 profit
        input_dim = first_batch_X.shape[1]
        
        if self.model is None:
            print(f"Initializing LogisticRegression with input_dim={input_dim}")
            self.model = LogisticRegression(input_dim).to(self.device)
            
        # 步骤 2：定义损失函数和优化器
        # BCELoss: 二元交叉熵损失函数，适用于二分类任务
        criterion = nn.BCELoss()
        
        # Adam 优化器：自适应学习率优化算法
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 步骤 3：创建 TensorBoard writer（如果启用）
        writer = None
        if self.tensorboard_enabled:
            writer = self._create_tensorboard_writer(
                experiment_name=self.experiment_name,
                log_dir=self.tensorboard_log_dir
            )
        
        # ⭐ 步骤 3.5：初始化 Best Model Checkpoint 追踪
        best_valid_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        
        print(f"Start training Torch Baseline (LR) on {self.device}...")
        if valid_loader is not None:
            print("✓ 启用 Best Model Checkpoint: 将自动保存验证集loss最低的模型")
        
        # 步骤 4：迭代训练
        for epoch in range(self.epochs):
            self.model.train()  # 设置为训练模式
            total_loss = 0
            
            # 批次训练循环
            for X_batch, y_batch, _ in train_loader:  # 解包三元组，忽略 profit
                # 将数据迁移到指定设备
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).float().view(-1, 1)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(X_batch)
                
                # 计算损失
                loss = criterion(outputs, y_batch)
                
                # 反向传播，计算梯度
                loss.backward()
                
                # 更新模型参数
                optimizer.step()
                
                total_loss += loss.item()
            
            # 计算平均训练 loss
            avg_loss = total_loss / len(train_loader)
            
            # 记录训练 loss 到 TensorBoard
            if writer:
                self._log_metrics(writer, {'Loss/train': avg_loss}, epoch)
            
            # 如果提供验证集，计算并记录验证 loss
            avg_valid_loss = None
            if valid_loader is not None:
                avg_valid_loss = self._compute_validation_loss(valid_loader, criterion)
                if writer:
                    self._log_metrics(writer, {'Loss/valid': avg_valid_loss}, epoch)
                
                # ⭐ Best Model Checkpoint: 如果当前验证loss是最低的，保存模型
                if avg_valid_loss < best_valid_loss:
                    best_valid_loss = avg_valid_loss
                    best_epoch = epoch + 1
                    import copy
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    
                    if (epoch + 1) % 5 == 0 or epoch < 10:
                        print(f"  ⭐ Epoch [{epoch+1}] - New best model! Valid Loss: {avg_valid_loss:.4f}")
            
            # 输出训练进度
            if (epoch + 1) % 5 == 0:
                valid_msg = f", Valid Loss: {avg_valid_loss:.4f}" if valid_loader else ""
                print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_loss:.4f}{valid_msg}")
        
        # ⭐ 步骤 4.5：恢复最佳模型
        if best_model_state is not None:
            print(f"\n✓ 训练完成！正在恢复 Best Model (Epoch {best_epoch}, Valid Loss: {best_valid_loss:.4f})...")
            self.model.load_state_dict(best_model_state)
            print(f"✓ 已恢复到效果最佳的模型参数（第 {best_epoch} 轮）")
        else:
            print(f"\n✓ 训练完成（未使用验证集，保存最后一轮的模型）")
        
        # 步骤 5：关闭 TensorBoard writer
        if writer:
            self._close_tensorboard_writer(writer)


    def predict(self, test_loader):
        """
        模型预测
        
        Args:
            test_loader: 测试数据的 DataLoader
            
        Returns:
            预测概率数组，shape 为 (n_samples,)
            
        Raises:
            ValueError: 若模型尚未训练或加载
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
            
        self.model.eval()  # 设置为评估模式
        preds = []
        
        # 禁用梯度计算，节省内存并加速推理
        with torch.no_grad():
            for X_batch, _, _ in test_loader:  # 解包三元组，忽略 y 和 profit
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                
                # 将预测结果转移至 CPU 并转换为 numpy 数组
                preds.extend(outputs.cpu().numpy().flatten())
                
        return np.array(preds)

    def save(self, path):
        """
        保存模型
        
        Args:
            path: 模型保存路径
        """
        # 保存模型参数、配置和输入维度
        torch.save({
            'state_dict': self.model.state_dict(),
            'config': self.config,
            'input_dim': self.model.linear.in_features
        }, path)

    def load(self, path):
        """
        加载模型
        
        Args:
            path: 模型文件路径
        """
        # map_location 确保跨设备兼容性（如 GPU 模型在 CPU 上加载）
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint.get('config', self.config)
        input_dim = checkpoint['input_dim']
        
        # 重建模型结构并加载参数
        self.model = LogisticRegression(input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
    
    def _compute_validation_loss(self, valid_loader, criterion):
        """
        计算验证集上的平均 loss
        
        Args:
            valid_loader: 验证数据的 DataLoader
            criterion: 损失函数
            
        Returns:
            验证集平均 loss
        """
        self.model.eval()  # 设置为评估模式
        valid_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch, _ in valid_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).float().view(-1, 1)
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                valid_loss += loss.item()
        
        self.model.train()  # 切回训练模式
        return valid_loss / len(valid_loader)

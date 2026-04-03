# -*- coding: utf-8 -*-
'''
@File    :   profit_resnet.py
@Time    :   2026/02/19 21:25:00
@Author  :   chensy 
@Desc    :   盈利敏感 ResNet 模型，使用 Rank-N-Contrast + MSE 双损失进行微调
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from .resnet_encoder import ResNetEncoder, RegressionHead
from .rank_n_contrast import RankNContrastLoss
from .base import BaseModel
from . import register_model


@register_model('profit_resnet')
class ProfitResNet(BaseModel):
    """
    盈利敏感 ResNet 模型
    
    加载预训练编码器，使用回归头预测盈利值，
    通过 MSE Loss + Rank-N-Contrast Loss 双损失联合优化
    """
    
    def __init__(self, config=None):
        """
        初始化盈利敏感模型
        
        Args:
            config (dict): 配置字典，包含：
                - lr: 学习率
                - epochs: 训练轮数
                - pretrained_encoder_path: 预训练编码器路径
                - params: 模型参数
                    - hidden_dim: 编码器隐藏维度
                    - dropout: Dropout 比例
                    - freeze_encoder: 是否冻结编码器
                    - lambda_rnc: Rank-N-Contrast 损失权重
                    - mse_weight: MSE 损失权重
                    - rnc_temperature: RnC 温度参数
        """
        self.config = config or {}
        
        # 通用参数
        self.lr = self.config.get('lr', 0.001)
        self.epochs = self.config.get('epochs', 50)
        self.pretrained_encoder_path = self.config.get('pretrained_encoder_path', None)
        
        # 模型特定参数
        params = self.config.get('params', {})
        self.hidden_dim = params.get('hidden_dim', 128)
        self.dropout = params.get('dropout', 0.3)
        self.freeze_encoder = params.get('freeze_encoder', False)
        
        # 损失函数参数
        self.lambda_rnc = params.get('lambda_rnc', 1.0)
        self.mse_weight = params.get('mse_weight', 1.0)
        self.rnc_temperature = params.get('rnc_temperature', 0.1)
        
        # 优化与验证选择策略
        self.weight_decay = params.get('weight_decay', 0.0)
        self.encoder_lr_scale = params.get('encoder_lr_scale', 1.0)
        self.head_lr_scale = params.get('head_lr_scale', 1.0)
        self.grad_clip_norm = params.get('grad_clip_norm', None)
        self.top_percent = params.get('top_percent', 0.3)
        
        print(f"盈利敏感 ResNet 配置:")
        print(f"  hidden_dim={self.hidden_dim}")
        print(f"  dropout={self.dropout}")
        print(f"  freeze_encoder={self.freeze_encoder}")
        print(f"  lambda_rnc={self.lambda_rnc}")
        print(f"  mse_weight={self.mse_weight}")
        print(f"  rnc_temperature={self.rnc_temperature}")
        print(f"  weight_decay={self.weight_decay}")
        print(f"  encoder_lr_scale={self.encoder_lr_scale}")
        print(f"  head_lr_scale={self.head_lr_scale}")
        print(f"  grad_clip_norm={self.grad_clip_norm}")
        print(f"  top_percent={self.top_percent}")
        
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型组件（延迟初始化）
        self.encoder = None
        self.regression_head = None
        self.model = None
    
    def _load_pretrained_encoder(self, input_dim):
        """
        加载预训练的编码器
        
        Args:
            input_dim (int): 输入维度
        """
        if self.pretrained_encoder_path and self.encoder is None:
            print(f"\n加载预训练编码器: {self.pretrained_encoder_path}")
            
            try:
                checkpoint = torch.load(self.pretrained_encoder_path, map_location=self.device)
                
                encoder_input_dim = checkpoint.get('input_dim', input_dim)
                encoder_hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
                encoder_dropout = checkpoint.get('dropout', self.dropout)
                
                self.encoder = ResNetEncoder(
                    input_dim=encoder_input_dim,
                    hidden_dim=encoder_hidden_dim,
                    dropout=encoder_dropout
                ).to(self.device)
                self.hidden_dim = encoder_hidden_dim
                
                self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                
                print(f"✓ 成功加载预训练编码器")
                print(f"  输入维度: {encoder_input_dim}")
                print(f"  隐藏维度: {encoder_hidden_dim}")
                
                if self.freeze_encoder:
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                    print(f"✓ 编码器参数已冻结")
                
            except Exception as e:
                print(f"⚠ 加载预训练编码器失败: {e}")
                print(f"   将使用随机初始化的编码器")
                self.encoder = None
    
    def train(self, train_loader, valid_loader=None):
        """
        盈利敏感有监督微调流程
        
        使用 MSE Loss + Rank-N-Contrast Loss 联合优化
        
        Args:
            train_loader: 训练数据加载器（返回 features, labels, profits）
            valid_loader: 验证数据加载器（可选）
        """
        # 步骤 1：推断输入维度
        first_batch_X, _, _ = next(iter(train_loader))
        input_dim = first_batch_X.shape[1]
        
        # 步骤 2：加载预训练编码器
        self._load_pretrained_encoder(input_dim)
        
        # 步骤 3：初始化编码器（如果未加载预训练权重）
        if self.encoder is None:
            print(f"\n使用随机初始化的编码器，输入维度={input_dim}")
            self.encoder = ResNetEncoder(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            ).to(self.device)
        
        # 步骤 4：添加回归头
        self.regression_head = RegressionHead(hidden_dim=self.hidden_dim).to(self.device)
        
        # 步骤 5：定义损失函数
        mse_criterion = nn.MSELoss()
        rnc_criterion = RankNContrastLoss(temperature=self.rnc_temperature)
        
        # 步骤 6：定义优化器
        optimizer = optim.AdamW(
            [
                {
                    'params': self.encoder.parameters(),
                    'lr': self.lr * self.encoder_lr_scale
                },
                {
                    'params': self.regression_head.parameters(),
                    'lr': self.lr * self.head_lr_scale
                }
            ],
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # 步骤 7：初始化 Best Model Checkpoint
        best_valid_profit = float('-inf')
        best_valid_loss = float('inf')
        best_epoch = 0
        best_encoder_state = None
        best_head_state = None
        
        print(f"\n开始盈利敏感微调，设备={self.device}")
        print(f"损失函数: {self.mse_weight} × MSE + {self.lambda_rnc} × RnC")
        print(
            f"优化器: AdamW(base_lr={self.lr}, enc_lr={self.lr * self.encoder_lr_scale}, "
            f"head_lr={self.lr * self.head_lr_scale}, weight_decay={self.weight_decay}, "
            f"grad_clip={self.grad_clip_norm})"
        )
        if valid_loader is not None:
            print(
                f"✓ 启用 Best Model Checkpoint: 优先保存验证集 Top{self.top_percent * 100:.0f}% "
                f"Profit 最高的模型，MSE 作为平手裁决"
            )
        
        # 步骤 9：训练循环
        for epoch in range(self.epochs):
            self.encoder.train()
            self.regression_head.train()
            
            total_loss = 0
            total_mse_loss = 0
            total_rnc_loss = 0
            
            for X_batch, _, profit_batch in train_loader:
                X_batch = X_batch.to(self.device)
                profit_batch = profit_batch.to(self.device).float().view(-1, 1)
                
                # 前向传播
                features = self.encoder(X_batch)
                y_pred = self.regression_head(features)
                
                # 计算双损失
                loss_mse = mse_criterion(y_pred, profit_batch)
                loss_rnc = rnc_criterion(features, profit_batch)
                
                loss = self.mse_weight * loss_mse + self.lambda_rnc * loss_rnc
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.regression_head.parameters()),
                        self.grad_clip_norm
                    )
                optimizer.step()
                
                total_loss += loss.item()
                total_mse_loss += loss_mse.item()
                total_rnc_loss += loss_rnc.item()
            
            # 计算平均损失
            n_batches = len(train_loader)
            avg_loss = total_loss / n_batches
            avg_mse = total_mse_loss / n_batches
            avg_rnc = total_rnc_loss / n_batches
            
            # 验证集评估
            avg_valid_loss = None
            avg_valid_profit = None
            valid_profit_metrics = None
            if valid_loader is not None:
                avg_valid_loss, valid_profit_metrics = self._evaluate_validation(
                    valid_loader,
                    mse_criterion
                )
                avg_valid_profit = valid_profit_metrics['total_profit']
                
                # Best Model Checkpoint
                if (
                    avg_valid_profit > best_valid_profit or
                    (
                        np.isclose(avg_valid_profit, best_valid_profit) and
                        avg_valid_loss < best_valid_loss
                    )
                ):
                    best_valid_profit = avg_valid_profit
                    best_valid_loss = avg_valid_loss
                    best_epoch = epoch + 1
                    best_encoder_state = copy.deepcopy(self.encoder.state_dict())
                    best_head_state = copy.deepcopy(self.regression_head.state_dict())
                    
                    if (epoch + 1) % 5 == 0 or epoch < 10:
                        print(
                            f"  ⭐ Epoch [{epoch+1}] - New best model! "
                            f"Valid Profit: {avg_valid_profit:.2f}, "
                            f"Valid MSE: {avg_valid_loss:.4f}"
                        )
            
            # 输出训练进度
            if (epoch + 1) % 5 == 0:
                valid_msg = ""
                if valid_loader is not None:
                    valid_msg = (
                        f", Valid MSE: {avg_valid_loss:.4f}, "
                        f"Valid Profit: {avg_valid_profit:.2f}, "
                        f"Valid Avg: {valid_profit_metrics['avg_profit']:.2f}"
                    )
                print(f"Epoch [{epoch+1}/{self.epochs}], "
                      f"Total: {avg_loss:.4f} (MSE: {avg_mse:.4f}, RnC: {avg_rnc:.4f}){valid_msg}")
        
        # 步骤 10：恢复最佳模型
        if best_encoder_state is not None:
            print(
                f"\n✓ 微调完成！正在恢复 Best Model (Epoch {best_epoch}, "
                f"Valid Profit: {best_valid_profit:.2f}, Valid MSE: {best_valid_loss:.4f})..."
            )
            self.encoder.load_state_dict(best_encoder_state)
            self.regression_head.load_state_dict(best_head_state)
            print(f"✓ 已恢复到效果最佳的模型参数（第 {best_epoch} 轮）")
        else:
            print(f"\n✓ 微调完成（未使用验证集，保存最后一轮的模型）")
        
        # 组装完整模型用于保存/预测
        self.model = nn.Sequential(self.encoder, self.regression_head)
    
    def predict(self, test_loader):
        """
        模型预测（输出盈利预测值）
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            np.ndarray: 预测盈利值数组
        """
        if self.encoder is None or self.regression_head is None:
            raise ValueError("模型尚未训练或加载！")
        
        self.encoder.eval()
        self.regression_head.eval()
        preds = []
        
        with torch.no_grad():
            for X_batch, _, _ in test_loader:
                X_batch = X_batch.to(self.device)
                features = self.encoder(X_batch)
                outputs = self.regression_head(features)
                preds.extend(outputs.cpu().numpy().flatten())
        
        return np.array(preds)
    
    def save(self, path):
        """保存模型"""
        if self.encoder is None or self.regression_head is None:
            raise ValueError("模型尚未训练！")
        
        input_dim = self.encoder.input_dim
        
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'regression_head_state_dict': self.regression_head.state_dict(),
            'config': self.config,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'input_dim': input_dim
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint.get('config', self.config)
        input_dim = checkpoint['input_dim']
        hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
        dropout = checkpoint.get('dropout', self.dropout)
        
        # 重建模型
        self.encoder = ResNetEncoder(input_dim, hidden_dim, dropout).to(self.device)
        self.regression_head = RegressionHead(hidden_dim).to(self.device)
        
        # 加载权重
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.regression_head.load_state_dict(checkpoint['regression_head_state_dict'])
        
        self.model = nn.Sequential(self.encoder, self.regression_head)
    
    def _compute_top_profit_metrics(self, preds, profits):
        """计算 Top N% 样本的累计利润指标。"""
        preds = np.array(preds)
        profits = np.array(profits)
        n_selected = int(len(preds) * self.top_percent)
        
        if n_selected <= 0:
            return {
                'total_profit': 0.0,
                'avg_profit': 0.0,
                'n_selected': 0
            }
        
        sorted_idx = np.argsort(preds)[::-1]
        top_idx = sorted_idx[:n_selected]
        selected_profits = profits[top_idx]
        return {
            'total_profit': float(selected_profits.sum()),
            'avg_profit': float(selected_profits.mean()),
            'n_selected': n_selected
        }
    
    def _evaluate_validation(self, valid_loader, criterion):
        """计算验证集 MSE 与 Top N% 利润指标。"""
        self.encoder.eval()
        self.regression_head.eval()
        valid_loss = 0
        valid_preds = []
        valid_profits = []
        
        with torch.no_grad():
            for X_batch, _, profit_batch in valid_loader:
                X_batch = X_batch.to(self.device)
                profit_batch = profit_batch.to(self.device).float().view(-1, 1)
                features = self.encoder(X_batch)
                outputs = self.regression_head(features)
                loss = criterion(outputs, profit_batch)
                valid_loss += loss.item()
                valid_preds.extend(outputs.cpu().numpy().flatten())
                valid_profits.extend(profit_batch.cpu().numpy().flatten())
        
        self.encoder.train()
        self.regression_head.train()
        avg_valid_loss = valid_loss / len(valid_loader)
        profit_metrics = self._compute_top_profit_metrics(valid_preds, valid_profits)
        return avg_valid_loss, profit_metrics

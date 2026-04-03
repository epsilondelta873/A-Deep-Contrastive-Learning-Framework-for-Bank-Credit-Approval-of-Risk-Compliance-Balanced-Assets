# -*- coding: utf-8 -*-
'''
@File    :   resnet.py
@Time    :   2026/02/01 16:42:00
@Author  :   chensy
@Desc    :   ResNet 消融实验实现
             与最终实验共享 ResNetEncoder 骨干，仅保留“无预训练、分类头”这一消融差异
'''

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from . import register_model
from .base import BaseModel
from .resnet_encoder import ResNetEncoder, ClassificationHead


class LegacyResidualBlock(nn.Module):
    """兼容旧版 checkpoint 的残差块实现。"""

    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu_final = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + identity
        return self.relu_final(out)


class LegacySimpleResNet(nn.Module):
    """兼容旧版 checkpoint 的单残差块 ResNet。"""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.residual_block = LegacyResidualBlock(hidden_dim, dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.residual_block(x)
        return self.output(x)


@register_model('resnet')
class ResNetModel(BaseModel):
    """
    ResNet 消融模型包装类

    该实现与最终实验共享：
    - `ResNetEncoder`
    - `ClassificationHead`
    - AdamW / 分组学习率 / 梯度裁剪
    - 基于验证集 Top-N% Profit 的最佳模型选择

    保留的消融差异：
    - 不使用对比学习预训练
    - 输出为分类概率而不是盈利回归值
    """

    def __init__(self, config=None):
        self.config = config or {}

        # 通用参数
        self.lr = self.config.get('lr', 0.001)
        self.epochs = self.config.get('epochs', 50)

        # 模型与优化器参数
        params = self.config.get('params', {})
        self.hidden_dim = params.get('hidden_dim', 128)
        self.dropout = params.get('dropout', 0.3)
        self.weight_decay = params.get('weight_decay', 0.0)
        self.encoder_lr_scale = params.get('encoder_lr_scale', 1.0)
        self.head_lr_scale = params.get('head_lr_scale', 1.0)
        self.grad_clip_norm = params.get('grad_clip_norm', None)

        # 评估口径
        evaluation_config = self.config.get('evaluation', {})
        self.top_percent = evaluation_config.get('top_percent', 0.3)
        self.threshold = evaluation_config.get('threshold', 0.2)

        print("ResNet 消融配置:")
        print(f"  hidden_dim={self.hidden_dim}")
        print(f"  dropout={self.dropout}")
        print(f"  weight_decay={self.weight_decay}")
        print(f"  encoder_lr_scale={self.encoder_lr_scale}")
        print(f"  head_lr_scale={self.head_lr_scale}")
        print(f"  grad_clip_norm={self.grad_clip_norm}")
        print(f"  top_percent={self.top_percent}")
        print(f"  threshold={self.threshold}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 延迟初始化
        self.encoder = None
        self.classifier = None
        self.model = None

    def _build_model(self, input_dim):
        """按共享骨干重建模型。"""
        self.encoder = ResNetEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)
        self.classifier = ClassificationHead(hidden_dim=self.hidden_dim).to(self.device)
        self.model = nn.Sequential(self.encoder, self.classifier)

    def train(self, train_loader, valid_loader=None):
        """
        单阶段监督分类训练。

        使用与最终实验一致的共享编码器骨干，但不加载预训练权重。
        """
        first_batch_X, _, _ = next(iter(train_loader))
        input_dim = first_batch_X.shape[1]

        if self.model is None:
            print(f"初始化 ResNet 编码器骨干，输入维度={input_dim}")
            self._build_model(input_dim)

        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            [
                {
                    'params': self.encoder.parameters(),
                    'lr': self.lr * self.encoder_lr_scale
                },
                {
                    'params': self.classifier.parameters(),
                    'lr': self.lr * self.head_lr_scale
                }
            ],
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        best_valid_profit = float('-inf')
        best_valid_loss = float('inf')
        best_epoch = 0
        best_encoder_state = None
        best_head_state = None

        print(f"开始训练 ResNet 消融模型，设备={self.device}...")
        print(
            f"优化器: AdamW(base_lr={self.lr}, enc_lr={self.lr * self.encoder_lr_scale}, "
            f"head_lr={self.lr * self.head_lr_scale}, weight_decay={self.weight_decay}, "
            f"grad_clip={self.grad_clip_norm})"
        )
        if valid_loader is not None:
            print(
                f"✓ 启用 Best Model Checkpoint: 优先保存验证集 Top{self.top_percent * 100:.0f}% "
                f"Profit 最高的模型，BCE 作为平手裁决"
            )

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch, _ in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).float().view(-1, 1)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()

                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.classifier.parameters()),
                        self.grad_clip_norm
                    )

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            avg_valid_loss = None
            valid_metrics = None
            if valid_loader is not None:
                avg_valid_loss, valid_metrics = self._evaluate_validation(valid_loader, criterion)

                if (
                    valid_metrics['total_profit'] > best_valid_profit or
                    (
                        np.isclose(valid_metrics['total_profit'], best_valid_profit) and
                        avg_valid_loss < best_valid_loss
                    )
                ):
                    best_valid_profit = valid_metrics['total_profit']
                    best_valid_loss = avg_valid_loss
                    best_epoch = epoch + 1
                    best_encoder_state = copy.deepcopy(self.encoder.state_dict())
                    best_head_state = copy.deepcopy(self.classifier.state_dict())

                    if (epoch + 1) % 5 == 0 or epoch < 10:
                        print(
                            f"  ⭐ Epoch [{epoch+1}] - New best model! "
                            f"Valid Profit: {valid_metrics['total_profit']:.2f}, "
                            f"Valid BCE: {avg_valid_loss:.4f}, "
                            f"Valid AUC: {self._format_auc(valid_metrics['auc'])}"
                        )

            if (epoch + 1) % 5 == 0:
                valid_msg = ""
                if valid_metrics is not None:
                    valid_msg = (
                        f", Valid BCE: {avg_valid_loss:.4f}, "
                        f"Valid Profit: {valid_metrics['total_profit']:.2f}, "
                        f"Valid Avg: {valid_metrics['avg_profit']:.2f}, "
                        f"Valid AUC: {self._format_auc(valid_metrics['auc'])}"
                    )
                print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_loss:.4f}{valid_msg}")

        if best_encoder_state is not None:
            print(
                f"\n✓ 训练完成！正在恢复 Best Model (Epoch {best_epoch}, "
                f"Valid Profit: {best_valid_profit:.2f}, Valid BCE: {best_valid_loss:.4f})..."
            )
            self.encoder.load_state_dict(best_encoder_state)
            self.classifier.load_state_dict(best_head_state)
            self.model = nn.Sequential(self.encoder, self.classifier)
            print(f"✓ 已恢复到效果最佳的模型参数（第 {best_epoch} 轮）")
        else:
            print(f"\n✓ 训练完成（未使用验证集，保存最后一轮的模型）")

    def predict(self, test_loader):
        """输出分类概率。"""
        if self.model is None:
            raise ValueError("模型尚未训练或加载！")

        self.model.eval()
        preds = []

        with torch.no_grad():
            for X_batch, _, _ in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                preds.extend(outputs.cpu().numpy().flatten())

        return np.array(preds)

    def save(self, path):
        """保存共享骨干版本的 checkpoint。"""
        if self.model is None:
            raise ValueError("模型尚未训练！")

        input_dim = self.encoder.input_dim

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'input_dim': input_dim,
            'format_version': 'shared_encoder_v1'
        }, path)

    def load(self, path):
        """
        加载模型。

        优先加载新的共享编码器格式，同时兼容旧版 `SimpleResNet` checkpoint。
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint.get('config', self.config)

        if 'model_state_dict' in checkpoint:
            input_dim = checkpoint['input_dim']
            self.hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
            self.dropout = checkpoint.get('dropout', self.dropout)
            self._build_model(input_dim)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            return

        if 'state_dict' in checkpoint:
            input_dim = checkpoint['input_dim']
            hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
            dropout = checkpoint.get('dropout', self.dropout)
            self.model = LegacySimpleResNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            ).to(self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            return

        raise ValueError("无法识别的 ResNet checkpoint 格式！")

    def _evaluate_validation(self, valid_loader, criterion):
        """按分类评估口径计算验证集 loss、AUC 与利润。"""
        self.model.eval()
        valid_loss = 0.0
        y_true_list = []
        y_pred_list = []
        profit_list = []

        with torch.no_grad():
            for X_batch, y_batch, profit_batch in valid_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).float().view(-1, 1)

                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                valid_loss += loss.item()

                y_true_list.append(y_batch.cpu().numpy().flatten())
                y_pred_list.append(outputs.cpu().numpy().flatten())
                profit_list.append(profit_batch.numpy().flatten())

        self.model.train()

        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        profits = np.concatenate(profit_list)

        metrics = self._calculate_profit_metrics(y_pred, profits)
        metrics['auc'] = self._safe_auc(y_true, y_pred)
        metrics['accuracy'] = float(((y_pred >= self.threshold).astype(int) == y_true).mean())

        return valid_loss / len(valid_loader), metrics

    def _calculate_profit_metrics(self, y_pred, profits):
        """与 evaluate.py 保持一致：分类模型按违约概率升序排序。"""
        n_samples = len(y_pred)
        n_selected = int(n_samples * self.top_percent)
        sorted_indices = np.argsort(y_pred)
        top_indices = sorted_indices[:n_selected]
        selected_profits = profits[top_indices]

        return {
            'total_profit': float(selected_profits.sum()),
            'avg_profit': float(selected_profits.mean()) if n_selected > 0 else 0.0,
            'selected_count': int(n_selected),
            'total_samples': int(n_samples)
        }

    @staticmethod
    def _safe_auc(y_true, y_pred):
        """当样本标签单一时，返回 NaN 而不是抛错。"""
        try:
            return float(roc_auc_score(y_true, y_pred))
        except ValueError:
            return float('nan')

    @staticmethod
    def _format_auc(value):
        """格式化 AUC 打印。"""
        if np.isnan(value):
            return "nan"
        return f"{value:.4f}"

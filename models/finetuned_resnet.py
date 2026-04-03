# -*- coding: utf-8 -*-
'''
@File    :   finetuned_resnet.py
@Time    :   2026/02/08 19:30:00
@Author  :   chensy
@Desc    :   对比学习后的分类微调 ResNet
             与最终实验共享编码器骨干和训练细节，仅保留分类头这一任务差异
'''

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from .resnet_encoder import ResNetEncoder, ClassificationHead
from .base import BaseModel
from . import register_model


@register_model('finetuned_resnet')
class FinetunedResNet(BaseModel):
    """
    微调 ResNet 模型

    加载预训练编码器，添加分类头，在有标签数据上进行有监督微调。
    与最终实验 `profit_resnet` 共享：
    - `ResNetEncoder`
    - AdamW / 分组学习率 / 梯度裁剪
    - 按验证集 Top-N% Profit 选择最佳 checkpoint
    """

    def __init__(self, config=None):
        self.config = config or {}

        # 通用参数
        self.lr = self.config.get('lr', 0.0001)
        self.epochs = self.config.get('epochs', 20)
        self.pretrained_encoder_path = self.config.get('pretrained_encoder_path', None)

        # 模型参数
        params = self.config.get('params', {})
        self.hidden_dim = params.get('hidden_dim', 128)
        self.dropout = params.get('dropout', 0.3)
        self.freeze_encoder = params.get('freeze_encoder', False)
        self.weight_decay = params.get('weight_decay', 0.0)
        self.encoder_lr_scale = params.get('encoder_lr_scale', 1.0)
        self.head_lr_scale = params.get('head_lr_scale', 1.0)
        self.grad_clip_norm = params.get('grad_clip_norm', None)

        # 评估口径
        evaluation_config = self.config.get('evaluation', {})
        self.top_percent = evaluation_config.get('top_percent', 0.3)
        self.threshold = evaluation_config.get('threshold', 0.2)

        print("微调 ResNet 配置:")
        print(f"  hidden_dim={self.hidden_dim}")
        print(f"  dropout={self.dropout}")
        print(f"  freeze_encoder={self.freeze_encoder}")
        print(f"  weight_decay={self.weight_decay}")
        print(f"  encoder_lr_scale={self.encoder_lr_scale}")
        print(f"  head_lr_scale={self.head_lr_scale}")
        print(f"  grad_clip_norm={self.grad_clip_norm}")
        print(f"  top_percent={self.top_percent}")
        print(f"  threshold={self.threshold}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模型组件（延迟初始化）
        self.encoder = None
        self.classifier = None
        self.model = None

    def _load_pretrained_encoder(self, input_dim):
        """加载预训练编码器。"""
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

                print("✓ 成功加载预训练编码器")
                print(f"  输入维度: {encoder_input_dim}")
                print(f"  隐藏维度: {encoder_hidden_dim}")

                if self.freeze_encoder:
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                    print("✓ 编码器参数已冻结")

            except Exception as e:
                print(f"⚠ 加载预训练编码器失败: {e}")
                print("   将使用随机初始化的编码器")
                self.encoder = None

    def train(self, train_loader, valid_loader=None):
        """有监督分类微调流程。"""
        first_batch_X, _, _ = next(iter(train_loader))
        input_dim = first_batch_X.shape[1]

        self._load_pretrained_encoder(input_dim)

        if self.encoder is None:
            print(f"\n使用随机初始化的编码器，输入维度={input_dim}")
            self.encoder = ResNetEncoder(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            ).to(self.device)

        self.classifier = ClassificationHead(hidden_dim=self.hidden_dim).to(self.device)
        self.model = nn.Sequential(self.encoder, self.classifier)

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
        best_model_state = None

        print(f"\n开始有监督微调，设备={self.device}")
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
                    best_model_state = copy.deepcopy(self.model.state_dict())

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

        if best_model_state is not None:
            print(
                f"\n✓ 微调完成！正在恢复 Best Model (Epoch {best_epoch}, "
                f"Valid Profit: {best_valid_profit:.2f}, Valid BCE: {best_valid_loss:.4f})..."
            )
            self.model.load_state_dict(best_model_state)
            print(f"✓ 已恢复到效果最佳的模型参数（第 {best_epoch} 轮）")
        else:
            print(f"\n✓ 微调完成（未使用验证集，保存最后一轮的模型）")

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
        """保存模型。"""
        if self.model is None:
            raise ValueError("模型尚未训练！")

        input_dim = self.encoder.input_dim

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'input_dim': input_dim
        }, path)

    def load(self, path):
        """加载模型。"""
        checkpoint = torch.load(path, map_location=self.device)

        self.config = checkpoint.get('config', self.config)
        input_dim = checkpoint['input_dim']
        hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
        dropout = checkpoint.get('dropout', self.dropout)

        self.encoder = ResNetEncoder(input_dim, hidden_dim, dropout).to(self.device)
        self.classifier = ClassificationHead(hidden_dim).to(self.device)
        self.model = nn.Sequential(self.encoder, self.classifier)
        self.model.load_state_dict(checkpoint['model_state_dict'])

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
        """标签单一时返回 NaN。"""
        try:
            return float(roc_auc_score(y_true, y_pred))
        except ValueError:
            return float('nan')

    @staticmethod
    def _format_auc(value):
        """格式化 AUC 用于打印。"""
        if np.isnan(value):
            return "nan"
        return f"{value:.4f}"

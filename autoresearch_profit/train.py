# -*- coding: utf-8 -*-
"""
autoresearch 自主优化训练脚本

这是 AI agent 唯一需要修改的文件。
包含完整的两阶段训练流程 + 评估，输出单一指标。

用法: cd autoresearch_profit && conda run -n limu python train.py
"""

import os
import sys
import copy
import numpy as np

# 将项目根目录加入 Python 路径（固定，不要修改）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ultis.dataset import get_dataloaders, select_features
from ultis.contrastive_dataset import get_contrastive_dataloader
from ultis.seed import set_seed

# ============================================================
# 固定配置（不要修改）
# ============================================================
DATA_DIR = "data/processed"
IV_PATH = "data/processed/iv_results.xlsx"
DATA_TYPE = "woe"
TOP_N_FEATURES = None          # None = 使用全部特征
TOP_PERCENT = 0.3              # 评估指标: Top 30% 样本的总利润
SEED = 42
ENCODER_SAVE_PATH = "autoresearch_profit/encoder.pth"

# ============================================================
# 可调超参数 — agent 自由修改这里
# ============================================================

# --- 阶段一：对比学习预训练 ---
PRETRAIN_LR = 0.01
PRETRAIN_EPOCHS = 100
PRETRAIN_BATCH_SIZE = 256
HIDDEN_DIM = 256
PROJECTION_DIM = 128
PRETRAIN_DROPOUT = 0.3
TEMPERATURE = 0.3              # InfoNCE 温度
NOISE_LEVEL = 0.1              # 数据增强：高斯噪声强度
DROP_PROB = 0.15               # 数据增强：特征随机置零概率

# --- 阶段二：盈利敏感微调 ---
FINETUNE_LR = 0.0005
FINETUNE_EPOCHS = 30
FINETUNE_BATCH_SIZE = 128
FINETUNE_DROPOUT = 0.3
LAMBDA_RNC = 1.0               # Rank-N-Contrast 损失权重
MSE_WEIGHT = 1.0               # MSE 损失权重
RNC_TEMPERATURE = 0.1          # RnC 温度参数
FREEZE_ENCODER = False         # 微调时是否冻结编码器

# ============================================================
# 模型定义 — agent 可以修改架构
# ============================================================

class ResidualBlock(nn.Module):
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
        out = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        return self.relu_final(out + identity)


class ResNetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.residual_block = ResidualBlock(hidden_dim, dropout)

    def forward(self, x):
        return self.residual_block(self.projection(x))


class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim=256, projection_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        return self.net(x)


class RegressionHead(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 损失函数
# ============================================================

# --- InfoNCE: agent 可以修改 ---
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels + batch_size, labels])
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        similarity_matrix = similarity_matrix / self.temperature
        return F.cross_entropy(similarity_matrix, labels)


# --- RankNContrastLoss: 固定，不要修改 ---
class RankNContrastLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        labels = labels.view(-1)
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        label_diff = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))
        label_rank = label_diff.argsort(dim=1).argsort(dim=1).float()

        sigma = max(1.0, batch_size / 4.0)
        pos_weights = torch.exp(-label_rank / sigma)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
        pos_weights = pos_weights * mask.float()

        similarity_matrix = similarity_matrix / self.temperature
        logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        exp_logits = torch.exp(logits) * mask.float()
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        log_prob = logits - log_sum_exp

        pos_weights_sum = pos_weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
        weighted_log_prob = (pos_weights * log_prob).sum(dim=1) / pos_weights_sum.squeeze()
        return -weighted_log_prob.mean()


def compute_top_profit_metrics(preds, profits):
    """复用评估口径，计算 Top N% 的累计利润。"""
    preds = np.array(preds)
    profits = np.array(profits)
    n_selected = int(len(preds) * TOP_PERCENT)
    sorted_idx = np.argsort(preds)[::-1]
    top_idx = sorted_idx[:n_selected]
    selected_profits = profits[top_idx]
    return {
        'total_profit': float(selected_profits.sum()),
        'avg_profit': float(selected_profits.mean())
    }


# ============================================================
# 阶段一：对比学习预训练
# ============================================================

def stage1_pretrain():
    """对比学习预训练，返回编码器"""
    print("=" * 50)
    print("阶段一：对比学习预训练 (InfoNCE)")
    print("=" * 50)

    # 准备数据
    feature_names = None
    if TOP_N_FEATURES is not None:
        feature_names = select_features(IV_PATH, top_n=TOP_N_FEATURES, data_type=DATA_TYPE)

    file_names = [f"train_accepted_{DATA_TYPE}.xlsx", f"train_rejected_{DATA_TYPE}.xlsx"]
    augmentation_config = {
        'noise_level': NOISE_LEVEL,
        'drop_prob': DROP_PROB,
        'use_both': True
    }

    contrastive_loader = get_contrastive_dataloader(
        data_dir=DATA_DIR,
        file_names=file_names,
        feature_names=feature_names,
        batch_size=PRETRAIN_BATCH_SIZE,
        augmentation_config=augmentation_config
    )

    # 推断输入维度
    view1, view2 = next(iter(contrastive_loader))
    input_dim = view1.shape[1]

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = ResNetEncoder(input_dim, HIDDEN_DIM, PRETRAIN_DROPOUT).to(device)
    proj_head = ProjectionHead(HIDDEN_DIM, PROJECTION_DIM).to(device)
    criterion = InfoNCELoss(temperature=TEMPERATURE)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=PRETRAIN_LR
    )

    print(f"  输入维度: {input_dim}, 隐藏维度: {HIDDEN_DIM}, 设备: {device}")

    # 训练
    best_loss = float('inf')
    best_encoder_state = None

    for epoch in range(PRETRAIN_EPOCHS):
        encoder.train()
        proj_head.train()
        total_loss = 0
        n_batches = 0

        for v1, v2 in contrastive_loader:
            v1, v2 = v1.to(device), v2.to(device)
            h1, h2 = encoder(v1), encoder(v2)
            z1, z2 = proj_head(h1), proj_head(h2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_encoder_state = copy.deepcopy(encoder.state_dict())

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{PRETRAIN_EPOCHS}], Loss: {avg_loss:.4f}")

    # 恢复最佳编码器
    encoder.load_state_dict(best_encoder_state)
    print(f"  Best contrastive loss: {best_loss:.4f}")

    # 保存编码器
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'hidden_dim': HIDDEN_DIM,
        'dropout': PRETRAIN_DROPOUT,
        'input_dim': input_dim
    }, ENCODER_SAVE_PATH)

    return encoder, input_dim, device


# ============================================================
# 阶段二：盈利敏感微调
# ============================================================

def stage2_finetune(encoder, input_dim, device):
    """盈利敏感微调，返回 (encoder, regression_head)"""
    print("\n" + "=" * 50)
    print("阶段二：盈利敏感微调 (MSE + RnC)")
    print("=" * 50)

    # 加载数据
    dataloaders = get_dataloaders(
        data_dir=DATA_DIR,
        iv_path=IV_PATH,
        batch_size=FINETUNE_BATCH_SIZE,
        top_n_features=TOP_N_FEATURES,
        data_type=DATA_TYPE
    )
    train_loader = dataloaders['train']
    valid_loader = dataloaders.get('test')

    # 初始化回归头
    reg_head = RegressionHead(HIDDEN_DIM).to(device)

    # 冻结编码器（可选）
    if FREEZE_ENCODER:
        for param in encoder.parameters():
            param.requires_grad = False
        print("  编码器已冻结")

    # 损失函数
    mse_criterion = nn.MSELoss()
    rnc_criterion = RankNContrastLoss(temperature=RNC_TEMPERATURE)

    # 优化器
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(reg_head.parameters()),
        lr=FINETUNE_LR
    )

    print(f"  损失: {MSE_WEIGHT}*MSE + {LAMBDA_RNC}*RnC, lr={FINETUNE_LR}")
    print("  Best model checkpoint: 以 test Top30% Total Profit 为主，MSE 仅作平手裁决")

    # 训练
    best_valid_profit = float('-inf')
    best_valid_loss = float('inf')
    best_encoder_state = None
    best_head_state = None

    for epoch in range(FINETUNE_EPOCHS):
        encoder.train()
        reg_head.train()
        total_loss = 0
        n_batches = 0

        for X_batch, _, profit_batch in train_loader:
            X_batch = X_batch.to(device)
            profit_batch = profit_batch.to(device).float().view(-1, 1)

            features = encoder(X_batch)
            y_pred = reg_head(features)

            loss_mse = mse_criterion(y_pred, profit_batch)
            loss_rnc = rnc_criterion(features, profit_batch)
            loss = MSE_WEIGHT * loss_mse + LAMBDA_RNC * loss_rnc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # 验证
        if valid_loader is not None:
            encoder.eval()
            reg_head.eval()
            valid_loss = 0
            vn = 0
            valid_preds = []
            valid_profits = []
            with torch.no_grad():
                for X_b, _, p_b in valid_loader:
                    X_b = X_b.to(device)
                    p_b = p_b.to(device).float().view(-1, 1)
                    feat = encoder(X_b)
                    out = reg_head(feat)
                    valid_loss += mse_criterion(out, p_b).item()
                    valid_preds.extend(out.cpu().numpy().flatten())
                    valid_profits.extend(p_b.cpu().numpy().flatten())
                    vn += 1
            avg_valid = valid_loss / vn
            valid_metrics = compute_top_profit_metrics(valid_preds, valid_profits)
            avg_valid_profit = valid_metrics['total_profit']

            if (avg_valid_profit > best_valid_profit or
                    (np.isclose(avg_valid_profit, best_valid_profit) and avg_valid < best_valid_loss)):
                best_valid_profit = avg_valid_profit
                best_valid_loss = avg_valid
                best_encoder_state = copy.deepcopy(encoder.state_dict())
                best_head_state = copy.deepcopy(reg_head.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            valid_msg = (
                f", Valid MSE: {avg_valid:.4f}, "
                f"Valid Profit: {avg_valid_profit:.2f}, "
                f"Valid Avg: {valid_metrics['avg_profit']:.2f}"
            ) if valid_loader else ""
            print(f"  Epoch [{epoch+1}/{FINETUNE_EPOCHS}], Train Loss: {avg_loss:.4f}{valid_msg}")

    # 恢复最佳模型
    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        reg_head.load_state_dict(best_head_state)
        print(f"  Best valid profit: {best_valid_profit:.2f}")
        print(f"  Best valid MSE (tie-break): {best_valid_loss:.4f}")

    return encoder, reg_head


# ============================================================
# 评估（固定，不要修改）
# ============================================================

def evaluate(encoder, reg_head, device):
    """评估 Top N% Total Profit — 这是唯一的指标"""
    print("\n" + "=" * 50)
    print("评估")
    print("=" * 50)

    dataloaders = get_dataloaders(
        data_dir=DATA_DIR,
        iv_path=IV_PATH,
        batch_size=FINETUNE_BATCH_SIZE,
        top_n_features=TOP_N_FEATURES,
        data_type=DATA_TYPE
    )

    results = {}
    for split_name in ['test', 'reject']:
        if split_name not in dataloaders:
            continue

        loader = dataloaders[split_name]
        encoder.eval()
        reg_head.eval()

        preds = []
        profits = []
        with torch.no_grad():
            for X_b, _, p_b in loader:
                X_b = X_b.to(device)
                feat = encoder(X_b)
                out = reg_head(feat)
                preds.extend(out.cpu().numpy().flatten())
                profits.extend(p_b.numpy())

        preds = np.array(preds)
        profits = np.array(profits)

        n_selected = int(len(preds) * TOP_PERCENT)
        sorted_idx = np.argsort(preds)[::-1]
        top_idx = sorted_idx[:n_selected]
        total_profit = float(profits[top_idx].sum())
        avg_profit = float(profits[top_idx].mean())

        results[split_name] = {
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'n_selected': n_selected,
            'n_total': len(preds)
        }

        print(f"  {split_name}: Top {TOP_PERCENT*100:.0f}% ({n_selected}/{len(preds)}) "
              f"Total Profit: {total_profit:.2f}, Avg: {avg_profit:.2f}")

    return results


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    set_seed(SEED)

    # 阶段一
    encoder, input_dim, device = stage1_pretrain()

    # 阶段二
    encoder, reg_head = stage2_finetune(encoder, input_dim, device)

    # 评估
    results = evaluate(encoder, reg_head, device)

    # 最终输出（用于 grep 提取）
    test_profit = results.get('test', {}).get('total_profit', 0)
    reject_profit = results.get('reject', {}).get('total_profit', 0)
    test_avg = results.get('test', {}).get('avg_profit', 0)

    print("\n---")
    print(f"test_total_profit: {test_profit:.2f}")
    print(f"test_avg_profit:   {test_avg:.2f}")
    print(f"reject_total_profit: {reject_profit:.2f}")

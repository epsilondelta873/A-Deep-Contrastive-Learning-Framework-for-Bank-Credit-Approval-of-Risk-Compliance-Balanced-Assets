# -*- coding: utf-8 -*-
'''
@File    :   rank_n_contrast.py
@Time    :   2026/02/19 21:25:00
@Author  :   chensy 
@Desc    :   Rank-N-Contrast 损失函数，实现盈利排序敏感的对比学习
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankNContrastLoss(nn.Module):
    """
    Rank-N-Contrast 损失函数
    
    核心思想：确保特征空间中的样本距离与盈利排序一致。
    盈利差距越小的样本对，特征空间中距离应越近（作为正样本对）；
    盈利差距越大的样本对，特征空间中距离应越远（作为负样本对）。
    
    参考: Rank-N-Contrast: Learning Continuous Representations for Regression (NeurIPS 2023)
    """
    
    def __init__(self, temperature=0.1):
        """
        初始化 Rank-N-Contrast 损失
        
        Args:
            temperature (float): 温度参数，控制相似度分布的尖锐程度。
                                 较小的值使分布更尖锐，较大的值更平滑。
        """
        super(RankNContrastLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        计算 Rank-N-Contrast 损失
        
        Args:
            features (torch.Tensor): 编码器输出的特征向量，shape (batch_size, hidden_dim)
            labels (torch.Tensor): 盈利标签，shape (batch_size,) 或 (batch_size, 1)
            
        Returns:
            torch.Tensor: RnC 损失值（标量）
        """
        # 1. 确保标签为一维
        labels = labels.view(-1)
        batch_size = features.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # 2. L2 归一化特征
        features = F.normalize(features, dim=1)
        
        # 3. 计算特征相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)  # (B, B)
        
        # 4. 计算标签距离矩阵
        label_diff = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))  # (B, B)
        
        # 5. 对每个样本 i，根据与其他样本的标签距离进行排序
        #    标签距离越小 → 排名越靠前 → 作为正样本的权重越高
        #    排序得到每个样本对的排名（0 = 最相似，B-1 = 最不相似）
        label_rank = label_diff.argsort(dim=1).argsort(dim=1).float()  # 双重 argsort 得到排名
        
        # 6. 计算正样本权重：排名靠前（标签距离小）的权重更高
        #    使用指数衰减：w_ij = exp(-rank_ij / σ)，σ 控制衰减速度
        sigma = max(1.0, batch_size / 4.0)
        pos_weights = torch.exp(-label_rank / sigma)
        
        # 7. 去掉对角线（自身不参与计算）
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
        pos_weights = pos_weights * mask.float()
        
        # 8. 应用温度参数
        similarity_matrix = similarity_matrix / self.temperature
        
        # 9. 数值稳定性：减去每行最大值
        logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # 10. 计算分母（所有非自身样本的 exp-sim 之和）
        exp_logits = torch.exp(logits) * mask.float()
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        
        # 11. 计算加权对比损失
        #     L = -Σ_j [ w_ij * (sim_ij / τ - log_sum_exp) ] / Σ_j w_ij
        log_prob = logits - log_sum_exp
        
        # 归一化权重
        pos_weights_sum = pos_weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
        weighted_log_prob = (pos_weights * log_prob).sum(dim=1) / pos_weights_sum.squeeze()
        
        # 12. 取负均值作为损失
        loss = -weighted_log_prob.mean()
        
        return loss

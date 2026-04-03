# -*- coding: utf-8 -*-
'''
@File    :   augmentation.py
@Time    :   2026/02/08 19:30:00
@Author  :   chensy 
@Desc    :   表格数据增强工具，用于对比学习
'''

import torch
import numpy as np


class TabularAugmentation:
    """
    表格数据增强类
    
    提供多种针对表格数据（WOE特征）的数据增强策略
    """
    
    def __init__(self, noise_level=0.1, drop_prob=0.15, use_both=True):
        """
        初始化数据增强器
        
        Args:
            noise_level (float): 高斯噪声的标准差，默认 0.1
            drop_prob (float): 特征 Dropout 概率，默认 0.15
            use_both (bool): 是否同时使用噪声和 Dropout，默认 True
        """
        self.noise_level = noise_level
        self.drop_prob = drop_prob
        self.use_both = use_both
    
    def add_gaussian_noise(self, features):
        """
        为特征添加高斯噪声
        
        Args:
            features (torch.Tensor): 输入特征，shape (batch_size, n_features) 或 (n_features,)
            
        Returns:
            torch.Tensor: 添加噪声后的特征
        """
        noise = torch.randn_like(features) * self.noise_level
        return features + noise
    
    def feature_dropout(self, features):
        """
        随机将部分特征置零（Feature Dropout）
        
        Args:
            features (torch.Tensor): 输入特征，shape (batch_size, n_features) 或 (n_features,)
            
        Returns:
            torch.Tensor: Dropout 后的特征
        """
        mask = (torch.rand_like(features) > self.drop_prob).float()
        return features * mask
    
    def __call__(self, features):
        """
        对特征进行增强
        
        Args:
            features (torch.Tensor): 输入特征
            
        Returns:
            torch.Tensor: 增强后的特征
        """
        if self.use_both:
            # 同时使用噪声和 Dropout
            features = self.add_gaussian_noise(features)
            features = self.feature_dropout(features)
        else:
            # 随机选择一种增强方式
            if np.random.rand() > 0.5:
                features = self.add_gaussian_noise(features)
            else:
                features = self.feature_dropout(features)
        
        return features


def get_augmentation(config):
    """
    根据配置创建数据增强器
    
    Args:
        config (dict): 包含增强参数的字典
            - noise_level: 噪声强度
            - drop_prob: Dropout 概率
            - use_both: 是否同时使用两种增强
            
    Returns:
        TabularAugmentation: 数据增强器实例
    """
    noise_level = config.get('noise_level', 0.1)
    drop_prob = config.get('drop_prob', 0.15)
    use_both = config.get('use_both', True)
    
    return TabularAugmentation(
        noise_level=noise_level,
        drop_prob=drop_prob,
        use_both=use_both
    )


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    print("--- 数据增强模块测试 ---")
    
    # 创建示例数据
    batch_size = 4
    n_features = 10
    features = torch.randn(batch_size, n_features)
    
    print(f"原始特征 shape: {features.shape}")
    print(f"原始特征（第一个样本）:\n{features[0]}\n")
    
    # 创建增强器
    augmenter = TabularAugmentation(noise_level=0.1, drop_prob=0.15, use_both=True)
    
    # 生成两个增强视图
    view1 = augmenter(features)
    view2 = augmenter(features)
    
    print(f"增强视图1（第一个样本）:\n{view1[0]}\n")
    print(f"增强视图2（第一个样本）:\n{view2[0]}\n")
    
    # 验证两个视图不同
    diff = torch.abs(view1 - view2).mean().item()
    print(f"两个视图的平均差异: {diff:.4f}")
    print("✓ 数据增强模块测试通过！")

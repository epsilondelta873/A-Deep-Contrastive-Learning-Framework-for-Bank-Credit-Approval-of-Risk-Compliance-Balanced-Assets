# -*- coding: utf-8 -*-
'''
@File    :   base.py
@Time    :   2026/01/17 20:47:50
@Author  :   chensy 
@Desc    :   模型基类定义，规范所有模型的统一接口
'''

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    所有模型的抽象基类
    
    该类定义了训练、预测、保存和加载的标准接口。
    所有具体模型（如 Baseline、DNN 等）必须继承此类并实现所有抽象方法。
    """
    
    @abstractmethod
    def __init__(self, config):
        """
        初始化模型
        
        Args:
            config: 配置字典，包含模型超参数
        """
        pass

    @abstractmethod
    def train(self, train_loader, valid_loader=None):
        """
        训练模型
        
        Args:
            train_loader: 训练数据的 DataLoader
            valid_loader: 验证数据的 DataLoader（可选）
        """
        pass

    @abstractmethod
    def predict(self, test_loader):
        """
        模型预测
        
        Args:
            test_loader: 测试数据的 DataLoader
            
        Returns:
            预测概率数组，shape 为 (n_samples,)
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        保存模型
        
        Args:
            path: 模型保存路径
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        加载模型
        
        Args:
            path: 模型文件路径
        """
        pass

# -*- coding: utf-8 -*-
'''
@File    :   base.py
@Time    :   2026/01/17 20:47:50
@Author  :   chensy 
@Desc    :   模型基类定义，规范所有模型的统一接口
'''

from abc import ABC, abstractmethod
import os
from datetime import datetime


class BaseModel(ABC):
    """
    所有模型的抽象基类
    
    该类定义了训练、预测、保存和加载的标准接口。
    所有具体模型（如 Baseline、DNN 等）必须继承此类并实现所有抽象方法。
    同时提供 TensorBoard 日志记录的辅助方法。
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

    # ==================== TensorBoard 辅助方法 ====================
    
    def _create_tensorboard_writer(self, experiment_name=None, log_dir='runs'):
        """
        创建 TensorBoard SummaryWriter
        
        Args:
            experiment_name: 实验名称，如果为 None 则自动生成
            log_dir: 日志保存目录
            
        Returns:
            SummaryWriter 对象
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("Warning: tensorboard not installed. Skipping TensorBoard logging.")
            return None
        
        # 生成实验目录名称
        if experiment_name is None:
            # 使用模型名称和时间戳作为默认实验名称
            model_name = getattr(self, '__class__').__name__.lower()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"{model_name}_{timestamp}"
        
        # 创建完整的日志路径
        log_path = os.path.join(log_dir, experiment_name)
        
        print(f"TensorBoard logs will be saved to: {log_path}")
        print(f"To view, run: tensorboard --logdir={log_dir}")
        
        return SummaryWriter(log_dir=log_path)
    
    def _log_metrics(self, writer, metrics_dict, step):
        """
        记录指标到 TensorBoard
        
        Args:
            writer: TensorBoard SummaryWriter 对象
            metrics_dict: 指标字典，格式为 {metric_name: value}
            step: 当前步数（通常是 epoch 数）
        """
        if writer is None:
            return
        
        for metric_name, value in metrics_dict.items():
            writer.add_scalar(metric_name, value, step)
    
    def _close_tensorboard_writer(self, writer):
        """
        关闭 TensorBoard SummaryWriter
        
        Args:
            writer: TensorBoard SummaryWriter 对象
        """
        if writer is not None:
            writer.close()

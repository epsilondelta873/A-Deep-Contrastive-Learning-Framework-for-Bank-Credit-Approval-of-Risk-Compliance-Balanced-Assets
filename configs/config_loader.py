# -*- coding: utf-8 -*-
'''
@File    :   config_loader.py
@Time    :   2026/01/17 22:09:00
@Author  :   chensy 
@Desc    :   配置加载器，支持 YAML 配置文件加载和命令行参数合并
'''

import os
import yaml
from typing import Dict, Any, Optional


class Config:
    """配置容器类，支持点语法访问配置项。
    
    该类将配置字典转换为支持点语法访问的对象，
    例如: config.model.lr 而不是 config['model']['lr']
    
    Example:
        >>> config = Config({'model': {'lr': 0.01}})
        >>> print(config.model.lr)  # 输出: 0.01
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """初始化配置对象。
        
        Args:
            config_dict: 配置字典
        """
        self._config = config_dict
        
        # 递归转换嵌套字典为 Config 对象
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        """支持字典式访问。
        
        Args:
            key: 配置键名
            
        Returns:
            配置值
        """
        return self._config[key]
    
    def __contains__(self, key):
        """支持 in 操作符。
        
        Args:
            key: 配置键名
            
        Returns:
            bool: 键是否存在
        """
        return key in self._config
    
    def get(self, key, default=None):
        """安全获取配置值。
        
        Args:
            key: 配置键名
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为普通字典。
        
        Returns:
            dict: 配置字典
        """
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self):
        """字符串表示。
        
        Returns:
            str: 配置的字符串表示
        """
        return f"Config({self._config})"


def load_config(config_path: str) -> Config:
    """加载 YAML 配置文件。
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Config: 配置对象
        
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML 格式错误
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        config_dict = {}
    
    return Config(config_dict)


def merge_args_with_config(config: Config, args) -> Config:
    """将命令行参数合并到配置中。
    
    命令行参数优先级高于配置文件。
    该函数会根据参数名自动映射到配置层级。
    
    映射规则：
    - model_name -> model.name
    - lr -> model.lr
    - epochs -> model.epochs
    - data_dir -> data.data_dir
    - iv_path -> data.iv_path
    - batch_size -> data.batch_size
    - top_n_features -> data.top_n_features
    - output_dir -> training.output_dir
    - top_percent -> evaluation.top_percent
    - model_type -> evaluation.model_type
    - output_file -> evaluation.output_file
    
    Args:
        config: 配置对象
        args: argparse.Namespace 对象
        
    Returns:
        Config: 合并后的配置对象
    """
    # 转换为字典便于修改
    config_dict = config.to_dict()
    
    # 定义参数映射关系
    arg_mapping = {
        # 模型参数
        'model_name': ('model', 'name'),
        'lr': ('model', 'lr'),
        'epochs': ('model', 'epochs'),
        
        # 数据参数
        'data_dir': ('data', 'data_dir'),
        'iv_path': ('data', 'iv_path'),
        'batch_size': ('data', 'batch_size'),
        'top_n_features': ('data', 'top_n_features'),
        'data_type': ('data', 'data_type'),
        
        # 训练参数
        'output_dir': ('training', 'output_dir'),
        
        # 评估参数
        'top_percent': ('evaluation', 'top_percent'),
        'model_type': ('evaluation', 'model_type'),
        'threshold': ('evaluation', 'threshold'),
        'output_file': ('evaluation', 'output_file'),
    }
    
    # 确保所有顶级键存在
    for top_key in ['model', 'data', 'training', 'evaluation']:
        if top_key not in config_dict:
            config_dict[top_key] = {}
    
    # 遍历命令行参数
    for arg_name, arg_value in vars(args).items():
        # 跳过 None 值（未指定的参数）
        if arg_value is None:
            continue
        
        # 跳过特殊参数
        if arg_name in ['config', 'model_path']:
            continue
        
        # 如果参数在映射表中，更新配置
        if arg_name in arg_mapping:
            top_key, sub_key = arg_mapping[arg_name]
            config_dict[top_key][sub_key] = arg_value
    
    return Config(config_dict)


def get_default_config_path() -> str:
    """获取默认配置文件路径。
    
    Returns:
        str: 默认配置文件的绝对路径
    """
    # 假设项目结构：project/configs/config_loader.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'base_config.yaml')
    return config_path


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    """测试配置加载器功能"""
    
    # 测试 1: 加载配置文件
    print("=== 测试 1: 加载配置文件 ===")
    config = load_config(get_default_config_path())
    print(f"模型名称: {config.model.name}")
    print(f"学习率: {config.model.lr}")
    print(f"批次大小: {config.data.batch_size}")
    
    # 测试 2: 字典访问
    print("\n=== 测试 2: 字典访问 ===")
    print(f"config['model']['lr'] = {config['model']['lr']}")
    
    # 测试 3: 转换为字典
    print("\n=== 测试 3: 转换为字典 ===")
    config_dict = config.to_dict()
    print(f"配置字典: {config_dict}")
    
    # 测试 4: 模拟命令行参数合并
    print("\n=== 测试 4: 命令行参数合并 ===")
    from argparse import Namespace
    args = Namespace(
        model_name='baseline',
        lr=0.001,  # 覆盖配置文件中的值
        epochs=30,  # 覆盖配置文件中的值
        batch_size=None,  # 不覆盖
        data_dir=None
    )
    merged_config = merge_args_with_config(config, args)
    print(f"合并后学习率: {merged_config.model.lr}")  # 应为 0.001
    print(f"合并后训练轮数: {merged_config.model.epochs}")  # 应为 30
    print(f"合并后批次大小: {merged_config.data.batch_size}")  # 应保持原值 64

# -*- coding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2026/01/16 21:52:02
@Author  :   chensy 
@Desc    :   模型注册中心，实现模型的动态注册与工厂模式
'''

from importlib import import_module

# 全局模型注册表，存储模型名称到模型类的映射
# 格式: {'baseline': BaselineCheck, 'model_name': ModelClass}
MODELS = {}


def register_model(name):
    """
    模型注册装饰器
    
    该装饰器用于将模型类注册到全局 MODELS 字典中，
    实现模型的自动发现和动态加载。
    
    Args:
        name: 模型的注册名称（字符串）
        
    Returns:
        decorator: 装饰器函数
        
    Example:
        >>> @register_model('my_model')
        >>> class MyModel(BaseModel):
        >>>     pass
    """
    def decorator(cls):
        MODELS[name] = cls
        return cls
    return decorator


def get_model(model_name, **kwargs):
    """
    模型工厂函数
    
    根据模型名称从注册表中获取对应的模型类并实例化。
    
    Args:
        model_name: 模型的注册名称
        **kwargs: 传递给模型构造函数的参数
        
    Returns:
        实例化后的模型对象
        
    Raises:
        ValueError: 若模型名称未注册
        
    Example:
        >>> model = get_model('baseline', config={'lr': 0.01})
    """
    if model_name not in MODELS:
        raise ValueError(
            f"Model '{model_name}' not found. "
            f"Available models: {list(MODELS.keys())}. \n"
            f"提示：请确保已导入模型定义文件，否则模型无法被注册。"
        )
    return MODELS[model_name](**kwargs)


# ========================================================
# 自动导入已实现的模型
# ========================================================
# 说明：
# Python 的装饰器只有在类定义被执行时才会触发。
# 因此必须在此处显式导入模型文件，确保 @register_model 被执行。
# 未来添加新模型时，也需要在此处添加相应的 import 语句。


def _optional_import(module_name):
    """
    可选导入模块。

    用于兼容仓库中仍保留配置文件、但源码实现暂未纳入版本控制的模型。
    如果模块文件缺失则静默跳过；如果模块存在但其内部导入失败，则继续抛出异常。
    """
    try:
        import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as exc:
        expected_names = {f"{__name__}.{module_name}", module_name}
        if exc.name not in expected_names:
            raise

# 已注册的模型
from . import baseline

# DNN 与 FT-Transformer 在当前仓库中仅保留了配置文件，源码可能不在版本控制内。
# 因此这里使用可选导入，避免影响其它已实现模型的训练和评估。
_optional_import('dnn')

# ResNet 模型（简单版本 - 单残差块）
# 使用残差连接改善深度网络训练
from . import resnet

# FT-Transformer 模型
_optional_import('ft_transformer')

# 对比学习 ResNet 模型
from . import contrastive_resnet

# 微调 ResNet 模型
from . import finetuned_resnet

# 盈利敏感 ResNet 模型
from . import profit_resnet

# XGBoost 模型
from . import xgboost_model

# LightGBM 模型
from . import lightgbm_model

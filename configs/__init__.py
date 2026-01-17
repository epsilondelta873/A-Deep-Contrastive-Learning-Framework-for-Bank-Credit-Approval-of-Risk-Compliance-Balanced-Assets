# -*- coding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2026/01/17 22:09:00
@Author  :   chensy 
@Desc    :   配置模块初始化
'''

from .config_loader import load_config, merge_args_with_config, Config

__all__ = ['load_config', 'merge_args_with_config', 'Config']

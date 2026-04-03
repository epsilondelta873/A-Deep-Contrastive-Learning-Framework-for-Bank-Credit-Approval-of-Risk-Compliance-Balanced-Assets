# -*- coding: utf-8 -*-
'''
@File    :   xgboost_model.py
@Time    :   2026/02/19 21:57:00
@Author  :   chensy 
@Desc    :   XGBoost 模型，适配 BaseModel 接口
'''

import numpy as np
import joblib
import xgboost as xgb
from .base import BaseModel
from . import register_model


@register_model('xgboost')
class XGBoostModel(BaseModel):
    """
    XGBoost 分类模型
    
    将树模型适配到 DataLoader 接口，支持 early_stopping
    """
    
    def __init__(self, config=None):
        """
        初始化 XGBoost 模型
        
        Args:
            config (dict): 配置字典，包含：
                - params: XGBoost 参数
                    - max_depth, learning_rate, n_estimators 等
                - epochs: 作为 n_estimators 使用
        """
        self.config = config or {}
        params = self.config.get('params', {})
        
        self.xgb_params = {
            'max_depth': params.get('max_depth', 6),
            'learning_rate': params.get('learning_rate', 0.1),
            'n_estimators': params.get('n_estimators', 200),
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'min_child_weight': params.get('min_child_weight', 5),
            'reg_alpha': params.get('reg_alpha', 0.1),
            'reg_lambda': params.get('reg_lambda', 1.0),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': self.config.get('seed', 42),
            'verbosity': 0,
        }
        
        self.early_stopping_rounds = params.get('early_stopping_rounds', 20)
        self.model = None
        
        print(f"XGBoost 配置:")
        print(f"  max_depth={self.xgb_params['max_depth']}")
        print(f"  learning_rate={self.xgb_params['learning_rate']}")
        print(f"  n_estimators={self.xgb_params['n_estimators']}")
        print(f"  subsample={self.xgb_params['subsample']}")
        print(f"  early_stopping={self.early_stopping_rounds}")
    
    def _loader_to_numpy(self, loader):
        """从 DataLoader 提取 NumPy 数据"""
        X_list, y_list, profit_list = [], [], []
        for X_batch, y_batch, profit_batch in loader:
            X_list.append(X_batch.numpy())
            y_list.append(y_batch.numpy())
            profit_list.append(profit_batch.numpy())
        return np.concatenate(X_list), np.concatenate(y_list), np.concatenate(profit_list)
    
    def train(self, train_loader, valid_loader=None):
        """
        训练 XGBoost 模型
        
        Args:
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器（用于 early_stopping）
        """
        print("\n提取训练数据...")
        X_train, y_train, _ = self._loader_to_numpy(train_loader)
        print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
        
        fit_params = {}
        
        if valid_loader is not None:
            X_valid, y_valid, _ = self._loader_to_numpy(valid_loader)
            print(f"  验证集: {X_valid.shape[0]} 样本")
            fit_params['eval_set'] = [(X_valid, y_valid)]
            fit_params['verbose'] = 20
            self.xgb_params['early_stopping_rounds'] = self.early_stopping_rounds
        
        print(f"\n开始训练 XGBoost...")
        self.model = xgb.XGBClassifier(**self.xgb_params)
        self.model.fit(X_train, y_train, **fit_params)
        
        print(f"\n✓ XGBoost 训练完成！")
        if hasattr(self.model, 'best_iteration'):
            print(f"  最佳迭代: {self.model.best_iteration}")
    
    def predict(self, test_loader):
        """
        模型预测（输出违约概率）
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            np.ndarray: 预测概率数组
        """
        X_test, _, _ = self._loader_to_numpy(test_loader)
        return self.model.predict_proba(X_test)[:, 1]
    
    def save(self, path):
        """保存模型"""
        joblib.dump(self.model, path)
    
    def load(self, path):
        """加载模型"""
        self.model = joblib.load(path)

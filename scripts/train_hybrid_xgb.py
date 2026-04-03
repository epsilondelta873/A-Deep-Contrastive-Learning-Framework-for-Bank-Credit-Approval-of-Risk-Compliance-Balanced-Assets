# -*- coding: utf-8 -*-
"""
@File    :   train_hybrid_xgb.py
@Author  :   chensy
@Desc    :   混合模型训练与评估脚本：加载导出的高维 Embeddings 特征，
             独立训练 XGBoost 决策森林模型，并计算 Profit 评估指标。
"""

import argparse
import os
import sys
import numpy as np

# 将项目主目录加入可执行路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import xgboost as xgb
except ImportError:
    print("Error: 环境中未找到 xgboost，请在命令行执行 'pip install xgboost'")
    sys.exit(1)

from configs import load_config, merge_args_with_config
from evaluate import calculate_classification_metrics, calculate_profit_metrics
from ultis.seed import set_seed


def load_embeddings(output_dir, split_name):
    """
    从 npz 加载保存的特征向量
    """
    path = os.path.join(output_dir, f"{split_name}_emb.npz")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path)
    return data['X_emb'], data['y'], data['profits']


def main():
    parser = argparse.ArgumentParser(description="混合模型级联评估脚本 (ResNet Encoder + XGBoost)")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/hybrid_xgb_config.yaml", 
        help="混合模型的配置路径"
    )
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置文件的随机种子")
    args = parser.parse_args()

    # 1. 加载参数配置
    print(f"Loading configuration from {args.config}...")
    try:
        config = load_config(args.config)
        config = merge_args_with_config(config, args)
    except FileNotFoundError:
        print(f"Error: 配置文件 {args.config} 未找到！")
        sys.exit(1)
        
    seed = args.seed if args.seed is not None else config.xgb.get('random_state', 42)
    set_seed(seed)
    
    output_dir = config.extract.output_dir
    
    # 2. 依次加载训练集、测试集、拒绝集的特征
    print("加载已抽取的 Embedding 特征空间数据...")
    X_train, y_train, profits_train = load_embeddings(output_dir, "train")
    if X_train is None:
        print(f"Error: 找不到训练集特征 {os.path.join(output_dir, 'train_emb.npz')}")
        print("请确认是否已经先执行完毕 extract_embeddings.py 脚本。")
        sys.exit(1)
        
    X_test, y_test, profits_test = load_embeddings(output_dir, "test")
    X_reject, y_reject, profits_reject = load_embeddings(output_dir, "reject")
    print(f"✓ 训练集装载完毕: X_train 维度为 {X_train.shape}")
    
    # 3. 按配置构建模型并执行训练
    print("\n" + "="*50)
    print("开始训练 XGBoost 决策树森林...")
    print("使用的核心 XGB Hyperparameters:")
    xgb_params = config.xgb.to_dict()
    xgb_params['random_state'] = seed  # 关键修复：向树引擎底层强势注入随机种
    
    for k, v in xgb_params.items():
        print(f"  {k}: {v}")
    
    # 使用 XGBClassifier
    clf = xgb.XGBClassifier(**xgb_params)
    clf.fit(X_train, y_train)
    print("✓ XGBoost 第二阶段拟合训练完成！")
    
    # 4. 执行多维度的业务指标测评逻辑
    print("\n" + "="*50)
    print("开始评估混合模型预测性能与收益贡献...")
    
    eval_sets = {
        'Test': (X_test, y_test, profits_test),
        'Reject': (X_reject, y_reject, profits_reject)
    }
    
    top_percent = config.evaluation.top_percent
    threshold = config.evaluation.threshold
    
    for split_name, (X, y, profits) in eval_sets.items():
        if X is None:
            continue
            
        print(f"\n==== Evaluating on {split_name} Set ====")
        
        # XGBoost 的 predict_proba 返回包含 N 个样本 各类别概率 的矩阵，正类概率在索引 1
        y_pred_prob = clf.predict_proba(X)[:, 1]
        
        # 4.1 传统学术分类指标计算
        cls_metrics = calculate_classification_metrics(y, y_pred_prob, threshold=threshold)
        print(f"  [基础分类能力 (Classification Metrics)]")
        print(f"    AUC:       {cls_metrics['AUC']:.4f}")
        print(f"    Accuracy:  {cls_metrics['Accuracy']:.4f}")
        print(f"    Precision: {cls_metrics['Precision']:.4f}")
        print(f"    Recall:    {cls_metrics['Recall']:.4f}")
        print(f"    F1 Score:  {cls_metrics['F1']:.4f}")
        
        # 4.2 业务核心利润指标计算
        profit_metrics = calculate_profit_metrics(
            y_pred_prob, 
            profits, 
            model_type=config.evaluation.model_type,
            top_percent=top_percent
        )
        print(f"  [业务截断收益 (Profit Metrics - Top {top_percent*100:.0f}%)]")
        print(f"    人群划选:   已选中 {profit_metrics['Selected_Count']} / {profit_metrics['Total_Samples']} 位业务样本")
        print(f"    预估总收益: {profit_metrics['Total_Profit']:.2f}")
        print(f"    被选均收益: {profit_metrics['Avg_Profit']:.2f}")

    print("\n" + "="*50)
    print("✓ 评估流程安全终结！推荐及时将实验收益指标与纯神经网络模型填表对比。")
    print("="*50)


if __name__ == "__main__":
    main()

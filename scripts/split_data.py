import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main():
    # 1. 定义路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_data_path = os.path.join(project_root, 'data', 'processed', 'p1_raw_data.xlsx')
    woe_data_path = os.path.join(project_root, 'data', 'processed', 'p1_woe_data.xlsx')
    iv_results_path = os.path.join(project_root, 'data', 'processed', 'iv_results.xlsx')
    
    # 输出路径
    output_dir = os.path.join(project_root, 'data', 'processed')
    
    # 2. 读取数据
    print("正在加载数据...")
    df_raw = pd.read_excel(raw_data_path)
    df_woe = pd.read_excel(woe_data_path)
    df_iv = pd.read_excel(iv_results_path)
    
    # 确保Raw和WoE数据的索引一致（通常假设行顺序对应）
    if len(df_raw) != len(df_woe):
        print("Error: Raw data 和 WoE data 行数不一致！")
        return

    # 3. 特征选择 (Top 20 IV)
    print("正在进行特征选择...")
    # 按IV降序排列
    df_iv_sorted = df_iv.sort_values(by='iv', ascending=False)
    # 取前20个变量名
    top_20_vars = df_iv_sorted.head(20)['variable'].tolist()
    
    # 映射到WoE变量名 (假设WoE变量名为 '原始变量名_woe')
    # 根据之前查看的cols.txt，WoE变量名确实是 xXX_woe
    # 需要确认 iv_results.xlsx 中的 variable 列也是 xXX 格式
    # 通常calculate_iv.py生成的 variable 就是原始列名
    top_20_woe_vars = [f"{var}_woe" for var in top_20_vars]
    
    print(f"Top 20 IV 变量: {top_20_vars}")
    print(f"对应的 WoE 变量: {top_20_woe_vars}")
    
    # 验证这些列是否都在 df_woe 中
    missing_cols = [col for col in top_20_woe_vars if col not in df_woe.columns]
    if missing_cols:
        print(f"Error: 以下WoE变量在数据集中未找到: {missing_cols}")
        return

    # 4. 拆分训练集和测试集 (70% Train, 30% Test)
    print("拆分训练集和测试集 (70/30)...")
    # 为了保持 Raw 和 WoE 的切分一致，我们可以先切分索引，或者同时切分
    # 这里我们使用索引来切分，这样可以方便地从两个从df中提取
    indices = np.arange(len(df_raw))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
    
    # 根据索引提取
    train_raw = df_raw.iloc[train_idx].copy()
    test_raw = df_raw.iloc[test_idx].copy()
    
    train_woe = df_woe.iloc[train_idx].copy()
    test_woe = df_woe.iloc[test_idx].copy()
    
    print(f"训练集大小: {len(train_raw)}, 测试集大小: {len(test_raw)}")
    
    # 5. 构建逻辑回归模型 (在训练集WoE上)
    print("正在构建逻辑回归模型...")
    X_train = train_woe[top_20_woe_vars]
    y_train = train_woe['y']
    
    # 处理可能的缺失值 (虽然calculate_iv应该处理了，但以防万一)
    if X_train.isnull().any().any():
        print("Warning: 训练数据中存在缺失值，正在使用均值填充...")
        X_train = X_train.fillna(X_train.mean())

    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # 6. 为训练集打分并生成拒绝集
    print("正在为训练集打分...")
    # 预测概率 (取为1的概率)
    train_scores = clf.predict_proba(X_train)[:, 1]
    
    # 将分数添加到dataframe以便排序
    # 注意：这里我们添加到一个临时列，以免影响原始数据结构，或者直接添加
    train_woe['score'] = train_scores
    train_raw['score'] = train_scores # 也要给raw加，方便对应
    
    # 升序排列 (分数低 -> 高)
    # 后20% (即分数最高的20%) 为拒绝集
    train_woe_sorted = train_woe.sort_values(by='score', ascending=True)
    train_raw_sorted = train_raw.sort_values(by='score', ascending=True)
    
    n_train = len(train_woe_sorted)
    cutoff_index = int(n_train * 0.8) # 80%的位置
    
    # 前80% -> 接受集 (Accepted)
    # 后20% -> 拒绝集 (Rejected)
    
    train_accepted_woe = train_woe_sorted.iloc[:cutoff_index]
    train_rejected_woe = train_woe_sorted.iloc[cutoff_index:]
    
    train_accepted_raw = train_raw_sorted.iloc[:cutoff_index]
    train_rejected_raw = train_raw_sorted.iloc[cutoff_index:]
    
    print(f"接受集大小: {len(train_accepted_woe)} (Top 80%)")
    print(f"拒绝集大小: {len(train_rejected_woe)} (Bottom 20%)")
    
    # 7. 导出结果
    print("正在导出文件...")
    
    # 为了保持整洁，导出前可以选择去掉临时添加的 'score' 列，或者保留供检查
    # 用户需求里没说要保留score，但通常保留会有用。
    # "需要导出的结果为... # 30%的测试集... # 训练集建模后的前80%... # 训练集建模后的后20%"
    # 我们保留score列吧，这对后续分析有帮助。
    
    # 测试集
    test_raw.to_excel(os.path.join(output_dir, 'test_raw.xlsx'), index=False)
    test_woe.to_excel(os.path.join(output_dir, 'test_woe.xlsx'), index=False)
    
    # 训练接受集
    train_accepted_raw.to_excel(os.path.join(output_dir, 'train_accepted_raw.xlsx'), index=False)
    train_accepted_woe.to_excel(os.path.join(output_dir, 'train_accepted_woe.xlsx'), index=False)
    
    # 训练拒绝集
    train_rejected_raw.to_excel(os.path.join(output_dir, 'train_rejected_raw.xlsx'), index=False)
    train_rejected_woe.to_excel(os.path.join(output_dir, 'train_rejected_woe.xlsx'), index=False)
    
    print("所有文件导出完成！")

if __name__ == "__main__":
    main()


import pandas as pd
import os
import glob

def verify():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    
    files_to_check = [
        'test_raw.xlsx', 'test_woe.xlsx',
        'train_accepted_raw.xlsx', 'train_accepted_woe.xlsx',
        'train_rejected_raw.xlsx', 'train_rejected_woe.xlsx'
    ]
    
    print("正在检查文件是否存在...")
    all_exist = True
    for f in files_to_check:
        path = os.path.join(processed_dir, f)
        if not os.path.exists(path):
            print(f"缺失: {f}")
            all_exist = False
        else:
            print(f"找到: {f}")
    
    if not all_exist:
        print("验证失败：缺少文件。")
        return

    print("\n正在检查数据一致性...")
    try:
        test_raw = pd.read_excel(os.path.join(processed_dir, 'test_raw.xlsx'))
        test_woe = pd.read_excel(os.path.join(processed_dir, 'test_woe.xlsx'))
        
        train_accepted_raw = pd.read_excel(os.path.join(processed_dir, 'train_accepted_raw.xlsx'))
        train_accepted_woe = pd.read_excel(os.path.join(processed_dir, 'train_accepted_woe.xlsx'))
        
        train_rejected_raw = pd.read_excel(os.path.join(processed_dir, 'train_rejected_raw.xlsx'))
        train_rejected_woe = pd.read_excel(os.path.join(processed_dir, 'train_rejected_woe.xlsx'))
        
        # 1. 形状检查
        if len(test_raw) != len(test_woe):
            print(f"失败：测试集长度不匹配: Raw={len(test_raw)}, WoE={len(test_woe)}")
        else:
            print(f"通过：测试集长度匹配 ({len(test_raw)})")
            
        if len(train_accepted_raw) != len(train_accepted_woe):
             print(f"失败：训练集接受样本长度不匹配")
        else:
             print(f"通过：训练集接受样本长度匹配 ({len(train_accepted_raw)})")

        if len(train_rejected_raw) != len(train_rejected_woe):
             print(f"失败：训练集拒绝样本长度不匹配")
        else:
             print(f"通过：训练集拒绝样本长度匹配 ({len(train_rejected_raw)})")
             
        total_train = len(train_accepted_raw) + len(train_rejected_raw)
        total_all = total_train + len(test_raw)
        train_ratio = total_train / total_all
        print(f"训练集比例: {train_ratio:.4f} (预期 ~0.7)")
        
        # 2. 拒绝集比例检查
        # Rejected (Back 20%) should refer to 20% of the TRAIN set.
        rejected_ratio = len(train_rejected_raw) / total_train
        print(f"训练集中拒绝集比例: {rejected_ratio:.4f} (预期 ~0.2)")
        
        # 3. 分数一致性检查
        # Accepted set scores should be <= Rejected set scores (because we sorted asc and took bottom 80% as accepted)
        # Note: 'score' column should be present if we exported it.
        if 'score' in train_accepted_raw.columns and 'score' in train_rejected_raw.columns:
            max_accepted_score = train_accepted_raw['score'].max()
            min_rejected_score = train_rejected_raw['score'].min()
            print(f"最大接受分数: {max_accepted_score}")
            print(f"最小拒绝分数: {min_rejected_score}")
            
            if max_accepted_score <= min_rejected_score:
                print("通过：分数分隔正确 (最大接受分数 <= 最小拒绝分数)")
            else:
                 print("失败：分数分隔不正确！")
        else:
            print("警告：未找到 'score' 列，跳过分数验证。")

    except Exception as e:
        print(f"验证失败，错误信息: {e}")

if __name__ == "__main__":
    verify()

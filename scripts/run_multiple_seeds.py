# -*- coding: utf-8 -*-
"""
@File    :   run_multiple_seeds.py
@Author  :   chensy
@Desc    :   通用的多次随机种子批处理调度总控外壳。
             支持三种执行模式：
             1. 单脚本模式（如混合树模型，训练+评估一体）
             2. 双脚本模式（train.py + evaluate.py）
             3. 三步流水线（pretrain → finetune → evaluate，用于对比学习系列）
             自动挂载 seed、执行编译、拦截控制台日志、解析指标落库以备 Std 计算。
"""

import argparse
import subprocess
import os
import re
import csv
from datetime import datetime

# 全局默认的一组高质量随机种子池（10 个），可通过 --seeds 参数覆盖
DEFAULT_SEED_POOL = [42, 1024, 2026, 777, 999, 3407, 1234, 2048, 314, 8888]

# 适配本项目两种打印习惯：
# 1. evaluate.py 中的英文 "Total Profit", "Evaluating on test set..."
# 2. train_hybrid_XXX.py 中的中文 "预估总收益", "Evaluating on Test Set"
METRIC_REGEXES = {
    'AUC': r'AUC:\s+([0-9\.]+)',
    'Accuracy': r'Accuracy:\s+([0-9\.]+)',
    'Total_Profit': r'(?:Total Profit|预估总收益):\s+([-0-9\.]+)',
    'Avg_Profit': r'(?:Avg Profit|被选均收益):\s+([-0-9\.]+)'
}

def parse_output(stdout_str):
    """从子进程日志中，精准截获并抽离各测试集合的数学指标。

    Args:
        stdout_str (str): 子进程的标准输出字符串。

    Returns:
        dict: 以数据集名称为 key、指标字典为 value 的嵌套字典。
    """
    results = {}
    
    # 支持匹配 "Evaluating on Test Set" 或 "Evaluating on test set..."
    splits = re.split(r'Evaluating on (\w+) [Ss]et', stdout_str)
    
    for i in range(1, len(splits), 2):
        # 统一转为首字母大写以便聚合时合并 (如: Test, Reject)
        split_name = splits[i].capitalize() 
        split_content = splits[i+1]
        
        split_metrics = {}
        for metric_name, regex in METRIC_REGEXES.items():
            match = re.search(regex, split_content)
            if match:
                split_metrics[metric_name] = float(match.group(1))
        
        if split_metrics:
            results[split_name] = split_metrics
            
    return results

def main():
    parser = argparse.ArgumentParser(description="项目通用架构：多次随机种子测试收集与方差记录器")
    parser.add_argument('--pretrain_script', type=str, default=None,
                        help="[可选] 预训练脚本路径（如 train_contrastive.py），用于对比学习三步流水线")
    parser.add_argument('--train_script', type=str, required=True,
                        help="要运行的主框架/训练脚本路径")
    parser.add_argument('--config', type=str, required=True,
                        help="要搭配的配置文件路径")
    parser.add_argument('--eval_script', type=str, default=None,
                        help="[可选] 若评估步骤是单独的脚本，请指定其路径（如 evaluate.py）")
    parser.add_argument('--model_path', type=str, default=None,
                        help="[可选] 传递给 evaluate.py 的 --model_path 参数")
    parser.add_argument('--model_name', type=str, default=None,
                        help="[可选] 覆盖 yaml 里的配置，用新网络名初始化模型评估器")
    parser.add_argument('--output', type=str, default="experiment_summary.csv",
                        help="指标追加汇总表名")
    parser.add_argument('--seeds', type=str, default=None,
                        help="自定义种子列表，逗号分隔（如 42,100,200）。不指定则使用默认 10 个种子")
    
    args = parser.parse_args()
    
    # 解析种子列表
    if args.seeds:
        seed_pool = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seed_pool = DEFAULT_SEED_POOL
    
    output_path = args.output
    file_exists = os.path.exists(output_path)
    
    csv_headers = [
        "Experiment_Time", "Algorithm", "Config", "Seed", 
        "Split", "AUC", "Accuracy", "Total_Profit", "Avg_Profit"
    ]
    
    with open(output_path, mode='a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        
        if not file_exists:
            writer.writeheader()
        
        print(f"\n🚀 开始对 [{args.train_script}] 进行多次定常种子交叉评估...")
        if args.pretrain_script:
            print(f"   预训练脚本: [{args.pretrain_script}]")
        print(f"验证方案队列 ({len(seed_pool)} 个种子): {seed_pool}")
        print("="*60)
        
        for i, seed in enumerate(seed_pool, 1):
            print(f"\n[任务 {i}/{len(seed_pool)}] ▶ 正在锁定环境 Seed: {seed} 进行拟合...")
            
            # 步骤 0（可选）：执行预训练（对比学习三步流水线）
            if args.pretrain_script:
                pretrain_cmd = ["python", args.pretrain_script, "--config", args.config, "--seed", str(seed)]
                print(f"  | 挂载执行（预训练）: {' '.join(pretrain_cmd)}")
                
                pretrain_process = subprocess.run(pretrain_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                if pretrain_process.returncode != 0:
                    print(f"❌ 警告：Seed {seed} 预训练阶段发生了致命异常！")
                    print(f"底层 STDERR: {pretrain_process.stderr}")
                    continue
                    
                print(f"  ✓ [预训练完成，编码器已保存]")
            
            # 步骤 1：开启训练挂载
            train_cmd = ["python", args.train_script, "--config", args.config, "--seed", str(seed)]
            print(f"  | 挂载执行: {' '.join(train_cmd)}")
            
            train_process = subprocess.run(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if train_process.returncode != 0:
                print(f"❌ 警告：Seed {seed} 运行时发生了致命异常退栈！")
                print(f"底层 STDERR: {train_process.stderr}")
                continue
                
            # 步骤 2：执行评估分级（如果是独立评估体系的代码架构）
            if args.eval_script:
                eval_cmd = ["python", args.eval_script, "--config", args.config]
                if args.model_path:
                    eval_cmd.extend(["--model_path", args.model_path])
                if args.model_name:
                    eval_cmd.extend(["--model_name", args.model_name])
                    
                print(f"  | 挂载执行（分离式验证）: {' '.join(eval_cmd)}")
                
                eval_process = subprocess.run(eval_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                if eval_process.returncode != 0:
                    print(f"❌ 警告：分离验证代码运行时发生了报错退栈！")
                    print(f"底层 STDERR: {eval_process.stderr}")
                    continue
                    
                target_stdout = eval_process.stdout
                print("  ✓ [分离式验证输出已收集完毕]")
            else:
                # 否则说明主脚本内部附带评估（例如混合树模型）
                target_stdout = train_process.stdout
                print("  ✓ [单脚本直出日志已收集完毕]")
                
            # 步骤 3：数据正则提鲜落库
            results = parse_output(target_stdout)
            
            if not results:
                print(f"❌ 警告：未能在此次实验日志中正则命中合规的数字指标！请核实在评测点是否打印了 'Evaluating...' 和数值。")
                continue
                
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 确定算法名称：优先使用预训练脚本名（如果有的话用 train_script 名）
            algorithm_name = os.path.basename(args.train_script).replace(".py", "")
            
            for split_name, metrics in results.items():
                row = {
                    "Experiment_Time": current_time,
                    "Algorithm": algorithm_name,
                    "Config": os.path.basename(args.config),
                    "Seed": seed,
                    "Split": split_name,
                    "AUC": metrics.get("AUC", ""),
                    "Accuracy": metrics.get("Accuracy", ""),
                    "Total_Profit": metrics.get("Total_Profit", ""),
                    "Avg_Profit": metrics.get("Avg_Profit", "")
                }
                writer.writerow(row)
                f.flush()
                
            print(f"  ⭐ 本轮次核检完成，Seed {seed} 捕获的信息已沉淀入公共表单。")

    print("\n" + "="*60)
    print(f"✅ [{args.train_script}] 队列多子采样运行顺利杀青！")
    print(f"建议马上使用: 'python scripts/aggregate_metrics.py --csv {args.output}' 获取含标准差的结论。")
    print("="*60)

if __name__ == "__main__":
    main()

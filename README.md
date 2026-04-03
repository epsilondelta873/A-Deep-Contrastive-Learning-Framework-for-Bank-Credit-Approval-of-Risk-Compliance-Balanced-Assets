# 面向风险合规均衡性资产的银行审批深度对比学习框架研究

本项目研究目标：在传统信用评分模型的基础上，引入对比学习预训练和盈利敏感微调，提升Top-K客户的召回收益。

> 公开版本说明
> 本仓库仅公开可复现代码、配置文件和目录约定，不包含真实数据、模型权重、自动搜索实验目录以及论文写作目录。

项目的核心业务指标是：

- `测试集Top 30% Total Profit`

分类模型同时关注：

- `测试集与人工拒绝集AUC`

---

## 公开仓库目录结构

```text
.
├── configs/            # 配置文件
├── models/             # 模型定义
├── ultis/              # 数据集、增强、随机种子等工具函数
├── scripts/            # 数据预处理、嵌入提取、批量实验脚本
├── train.py            # 通用训练入口
├── evaluate.py         # 统一评估入口
├── notebook/           # 辅助分析脚本与 notebook
├── data/
│   ├── raw/            # 原始数据占位目录，不上传真实文件
│   ├── processed/      # 预处理数据占位目录，不上传真实文件
│   └── embeddings/     # 嵌入特征占位目录，不上传真实文件
└── model_param/        # 训练得到的权重和模型文件，不上传
```

公开仓库默认不提供以下内容：

- 原始数据和处理后数据
- 训练好的模型权重与编码器参数
- `autoresearch*` 本地自动搜索/试验目录
- `document/` 论文写作与办公材料目录
- 实验汇总表和其他本地生成文件

---

## 1. 项目包含的 8 个实验

### 消融实验

1. `base_config.yaml`
   - 基准模型：逻辑回归

2. `resnet_config.yaml`
   - 实验 A：ResNet 编码器

3. `contrastive_resnet_config.yaml`
   - 实验 B：ResNet + 无监督对比学习预训练 + 分类微调

4. `profit_resnet_config.yaml`
   - 实验 C：ResNet + 无监督对比学习预训练 + 盈利敏感微调

### 横向对比实验

5. `lightgbm_config.yaml`
   - LightGBM

6. `xgboost_config.yaml`
   - XGBoost

7. `hybrid_lgbm_config.yaml`
   - 编码器表征 + LightGBM

8. `hybrid_xgb_config.yaml`
   - 编码器表征 + XGBoost

---

## 2. 数据说明

训练和评估默认使用 `data/processed/` 下的 Excel 文件。公开仓库只保留目录定义，不附带真实数据；如需复现实验，请先在本地补齐以下文件：

- `train_accepted_woe.xlsx`
- `train_rejected_woe.xlsx`
- `test_woe.xlsx`
- `iv_results.xlsx`

如果需要重新从原始数据开始处理，可参考 `scripts/` 下的数据预处理脚本：

- `scripts/clean_raw_data.py`
- `scripts/split_data.py`
- `scripts/calculate_iv.py`
- `scripts/calculate_vif.py`

---

## 3. 如何训练

### 3.1 基准模型

```bash
python train.py --config configs/base_config.yaml
```

### 3.2 实验A：ResNet

```bash
python train.py --config configs/resnet_config.yaml
```

### 3.3 实验B：无监督对比学习预训练 + 分类微调

先做预训练：

```bash
python train_contrastive.py --config configs/contrastive_resnet_config.yaml
```

再做分类微调：

```bash
python train_finetune.py --config configs/contrastive_resnet_config.yaml
```

### 3.4 实验C：无监督对比学习预训练 + 盈利敏感微调

先做预训练：

```bash
python train_contrastive.py --config configs/profit_resnet_config.yaml
```

再做盈利敏感微调：

```bash
python train_profit_finetune.py --config configs/profit_resnet_config.yaml
```

### 3.5 LightGBM

```bash
python train.py --config configs/lightgbm_config.yaml
```

### 3.6 XGBoost

```bash
python train.py --config configs/xgboost_config.yaml
```

### 3.7 编码器表征 + LightGBM

先准备预训练编码器：

```bash
python train_contrastive.py --config configs/contrastive_resnet_config.yaml
```

提取表征：

```bash
python scripts/extract_embeddings.py --config configs/hybrid_lgbm_config.yaml
```

训练混合模型：

```bash
python scripts/train_hybrid_lgbm.py --config configs/hybrid_lgbm_config.yaml
```

### 3.8 编码器表征 + XGBoost

先准备预训练编码器：

```bash
python train_contrastive.py --config configs/contrastive_resnet_config.yaml
```

提取表征：

```bash
python scripts/extract_embeddings.py --config configs/hybrid_xgb_config.yaml
```

训练混合模型：

```bash
python scripts/train_hybrid_xgb.py --config configs/hybrid_xgb_config.yaml
```

---

## 4. 如何评估

以下评估命令都要求你本地已经存在对应的模型文件；公开仓库不附带任何训练好的权重。

### 4.1 基准模型

```bash
python evaluate.py \
  --config configs/base_config.yaml \
  --model_name baseline \
  --model_type classification \
  --model_path model_param/baseline_model.pkl
```

### 4.2 实验A：ResNet

```bash
python evaluate.py \
  --config configs/resnet_config.yaml \
  --model_name resnet \
  --model_type classification \
  --model_path model_param/resnet_model.pkl
```

### 4.3 实验B：无监督对比学习预训练 + 分类微调

```bash
python evaluate.py \
  --config configs/contrastive_resnet_config.yaml \
  --model_name finetuned_resnet \
  --model_type classification \
  --model_path model_param/finetuned_resnet_model.pkl
```

### 4.4 实验C：盈利敏感模型

```bash
python evaluate.py \
  --config configs/profit_resnet_config.yaml \
  --model_name profit_resnet \
  --model_type regression \
  --model_path model_param/profit_resnet_model.pkl
```

### 4.5 LightGBM

```bash
python evaluate.py \
  --config configs/lightgbm_config.yaml \
  --model_name lightgbm \
  --model_type classification \
  --model_path model_param/lightgbm_model.pkl
```

### 4.6 XGBoost

```bash
python evaluate.py \
  --config configs/xgboost_config.yaml \
  --model_name xgboost \
  --model_type classification \
  --model_path model_param/xgboost_model.pkl
```

### 4.7 混合模型

混合模型的评估已经集成在训练脚本里：

- `scripts/train_hybrid_lgbm.py`
- `scripts/train_hybrid_xgb.py`

运行训练脚本后会直接输出测试集和拒绝集指标。

---

## 5. 模型稳定性测试：10 个随机种子实验

模型稳定性测试统一使用：

```bash
python scripts/run_multiple_seeds.py ...
```

默认的 10 个随机种子为：

```text
42, 1024, 2026, 777, 999, 3407, 1234, 2048, 314, 8888
```

### 5.1 单阶段模型示例：基准模型

```bash
python scripts/run_multiple_seeds.py \
  --train_script train.py \
  --eval_script evaluate.py \
  --config configs/base_config.yaml \
  --model_path model_param/baseline_model.pkl \
  --model_name baseline \
  --output experiment_summary.csv
```

同理：

- `实验A：resnet_config.yaml`
- `LightGBM:lightgbm_config.yaml`
- `XGBoost:xgboost_config.yaml`

都可以用同样的模式，只需要替换配置文件和模型名。

### 5.2 两阶段分类模型示例：对比学习 + 分类微调

```bash
python scripts/run_multiple_seeds.py \
  --pretrain_script train_contrastive.py \
  --train_script train_finetune.py \
  --eval_script evaluate.py \
  --config configs/contrastive_resnet_config.yaml \
  --model_path model_param/finetuned_resnet_model.pkl \
  --model_name finetuned_resnet \
  --output experiment_summary.csv
```

### 5.3 两阶段盈利敏感模型示例

```bash
python scripts/run_multiple_seeds.py \
  --pretrain_script train_contrastive.py \
  --train_script train_profit_finetune.py \
  --eval_script evaluate.py \
  --config configs/profit_resnet_config.yaml \
  --model_path model_param/profit_resnet_model.pkl \
  --model_name profit_resnet \
  --output experiment_summary.csv
```

运行结束后，所有 seed 的原始结果会追加保存到：

- `experiment_summary.csv`

说明：

- `scripts/run_multiple_seeds.py` 目前最适合单阶段模型和“两阶段神经网络模型”。
- 对于混合模型（`hybrid_lgbm` / `hybrid_xgb`），如果要做 10 个随机种子测试，建议按同一 seed 依次重复执行：
  - `train_contrastive.py`
  - `scripts/extract_embeddings.py --seed <seed>`
  - `scripts/train_hybrid_lgbm.py --seed <seed>` 或 `scripts/train_hybrid_xgb.py --seed <seed>`

---

## 6. 实验结果汇总

将 `experiment_summary.csv` 聚合为“每组实验一行”的结果表：

```bash
python scripts/aggregate_metrics.py \
  --csv experiment_summary.csv \
  --output experiment_summary_aggregated.csv
```

输出文件：

- `experiment_summary_aggregated.csv`

当前聚合结果只保留：

- `AUC_mean`
- `AUC_std`
- `Total_Profit_mean`
- `Total_Profit_std`

并且按照论文中的实验顺序输出：

1. `base_config.yaml`
2. `resnet_config.yaml`
3. `contrastive_resnet_config.yaml`
4. `profit_resnet_config.yaml`
5. `lightgbm_config.yaml`
6. `xgboost_config.yaml`
7. `hybrid_lgbm_config.yaml`
8. `hybrid_xgb_config.yaml`

---

## 7. 代码结构

主要目录如下：

```text
configs/    # 配置文件
models/     # 模型实现
scripts/    # 批处理、多种子、混合模型、聚合脚本
ultis/      # 数据加载、增强、随机种子等工具
```

主要入口如下：

- `train.py`
- `evaluate.py`
- `train_contrastive.py`
- `train_finetune.py`
- `train_profit_finetune.py`
- `scripts/run_multiple_seeds.py`
- `scripts/aggregate_metrics.py`

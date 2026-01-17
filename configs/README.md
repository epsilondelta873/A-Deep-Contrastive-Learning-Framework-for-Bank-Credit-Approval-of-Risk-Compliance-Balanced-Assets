# 配置管理系统使用指南

本项目采用 YAML 配置文件管理所有超参数，支持命令行参数覆盖。

## 配置文件结构

配置文件位于 `configs/base_config.yaml`，包含以下部分：

```yaml
model:          # 模型相关配置
  name: baseline
  lr: 0.01
  epochs: 20

data:           # 数据相关配置
  data_dir: data/processed
  iv_path: data/processed/iv_results.xlsx
  batch_size: 64
  top_n_features: 20
  data_type: woe

training:       # 训练相关配置
  output_dir: checkpoints

evaluation:     # 评估相关配置
  top_percent: 0.3
  model_type: classification
  output_file: evaluation_results.csv
```

## 使用方法

### 1. 使用默认配置训练模型

```bash
python train.py
```

配置系统会自动加载 `configs/base_config.yaml` 中的默认参数。

### 2. 使用自定义配置文件

```bash
python train.py --config configs/my_experiment.yaml
```

### 3. 命令行参数覆盖配置文件

命令行参数优先级高于配置文件，可用于临时调整参数：

```bash
# 覆盖学习率和训练轮数
python train.py --lr 0.001 --epochs 30

# 覆盖批次大小和特征数量
python train.py --batch_size 128 --top_n_features 30
```

### 4. 组合使用

```bash
# 使用自定义配置文件，并覆盖部分参数
python train.py --config configs/experiment1.yaml --lr 0.005
```

## 评估模型

### 基本用法

```bash
python evaluate.py --model_path model_param/baseline_model.pkl
```

### 覆盖配置

```bash
# 修改评估的 top 百分比
python evaluate.py --model_path checkpoints/baseline_model.pkl --top_percent 0.2

# 指定模型类型
python evaluate.py --model_path checkpoints/baseline_model.pkl --model_type regression
```

## 创建新的实验配置

为不同实验创建独立的配置文件：

```bash
# 复制基础配置
cp configs/base_config.yaml configs/experiment1.yaml

# 编辑配置文件
# vim configs/experiment1.yaml

# 使用新配置运行实验
python train.py --config configs/experiment1.yaml
```

## 参数优先级

1. **最高优先级**: 命令行参数
2. **中等优先级**: 指定的配置文件（`--config`）
3. **最低优先级**: 默认配置文件（`configs/base_config.yaml`）

## 配置参数映射

命令行参数到配置文件的映射关系：

| 命令行参数         | 配置文件路径             | 说明       |
| ------------------ | ------------------------ | ---------- |
| `--model_name`     | `model.name`             | 模型名称   |
| `--lr`             | `model.lr`               | 学习率     |
| `--epochs`         | `model.epochs`           | 训练轮数   |
| `--batch_size`     | `data.batch_size`        | 批次大小   |
| `--top_n_features` | `data.top_n_features`    | 特征数量   |
| `--data_dir`       | `data.data_dir`          | 数据目录   |
| `--iv_path`        | `data.iv_path`           | IV文件路径 |
| `--output_dir`     | `training.output_dir`    | 输出目录   |
| `--top_percent`    | `evaluation.top_percent` | Top百分比  |
| `--model_type`     | `evaluation.model_type`  | 模型类型   |
| `--output_file`    | `evaluation.output_file` | 输出文件   |

## 完整示例

```bash
# 示例 1: 使用默认配置训练 baseline 模型
python train.py

# 示例 2: 调整超参数训练
python train.py --lr 0.005 --epochs 50 --batch_size 128

# 示例 3: 使用实验配置文件
python train.py --config configs/experiment1.yaml

# 示例 4: 评估模型
python evaluate.py --model_path checkpoints/baseline_model.pkl

# 示例 5: 评估时修改参数
python evaluate.py --model_path checkpoints/baseline_model.pkl \
  --top_percent 0.2 --batch_size 128
```

## 最佳实践

1. **实验记录**: 为每个重要实验创建独立的配置文件
2. **版本控制**: 将配置文件纳入 Git 版本控制
3. **命名规范**: 配置文件使用描述性名称（如 `baseline_lr0.001.yaml`）
4. **参数追踪**: 训练开始时会打印当前使用的所有配置参数，便于记录

## 依赖安装

配置系统需要 PyYAML 库：

```bash
pip install pyyaml
```

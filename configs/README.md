# 配置管理系统完整指南

> **文档概要**  
> 本文档是项目配置系统的完整使用指南，涵盖以下内容：
> 1. **配置文件结构** - YAML 配置文件的组织方式和字段说明
> 2. **基本使用** - 训练、评估的常用命令和参数覆盖方法
> 3. **模型特定参数** - 如何为不同模型配置专属参数（`model.params`）
> 4. **新模型开发** - 从零开始创建新模型的完整流程
> 5. **最佳实践** - 实验管理、参数追踪等实用建议

---

## 📋 目录

- [快速开始](#快速开始)
- [配置文件结构](#配置文件结构)
- [基本使用方法](#基本使用方法)
- [模型特定参数 (model.params)](#模型特定参数-modelparams)
- [开发新模型完整流程](#开发新模型完整流程)
- [配置参数参考](#配置参数参考)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

---

## 🚀 快速开始

### 训练模型

```bash
# 使用默认配置训练 baseline 模型
python train.py

# 使用自定义配置训练 DNN 模型
python train.py --config configs/dnn_config.yaml

# 临时覆盖学习率和训练轮数
python train.py --lr 0.001 --epochs 30
```

### 评估模型

```bash
# 评估训练好的模型
python evaluate.py --model_path model_param/baseline_model.pkl

# 指定配置文件评估
python evaluate.py --model_path model_param/dnn_model.pkl \
                   --config configs/dnn_config.yaml
```

---

## 📁 配置文件结构

配置文件采用 YAML 格式，默认位于 `configs/base_config.yaml`，包含四个主要部分：

```yaml
# ====================================
# 模型配置
# ====================================
model:
  name: baseline          # 模型名称
  lr: 0.01               # 学习率（通用参数）
  epochs: 20             # 训练轮数（通用参数）
  params: {}             # 模型特定参数（各模型自定义）

# ====================================
# 数据配置
# ====================================
data:
  data_dir: data/processed                    # 数据目录路径
  iv_path: data/processed/iv_results.xlsx     # IV 结果文件路径
  batch_size: 64                              # 批次大小
  top_n_features: 20                          # 选择 IV 值最高的前 N 个特征
  data_type: woe                              # 数据类型: 'woe' 或 'raw'

# ====================================
# 训练配置
# ====================================
training:
  output_dir: model_param                    # 模型保存目录
  tensorboard:
    enabled: true                            # 是否启用 TensorBoard
    log_dir: runs                            # TensorBoard 日志保存目录
    log_interval: 1                          # 每隔多少个 epoch 记录一次

# ====================================
# 评估配置
# ====================================
evaluation:
  top_percent: 0.3                            # 选择排名前 N% 的样本计算利润
  model_type: classification                  # 模型类型: 'classification' 或 'regression'
  threshold: 0.2                              # 分类阈值（仅二分类模型使用）
  output_file: evaluation_results.csv         # 评估结果输出文件
```

---

## 💻 基本使用方法

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

### 5. 评估模型

```bash
# 基本用法
python evaluate.py --model_path model_param/baseline_model.pkl

# 修改评估的 top 百分比
python evaluate.py --model_path model_param/baseline_model.pkl --top_percent 0.2

# 指定模型类型
python evaluate.py --model_path model_param/baseline_model.pkl --model_type regression
```

### 参数优先级

1. **最高优先级**: 命令行参数
2. **中等优先级**: 指定的配置文件（`--config`）
3. **最低优先级**: 默认配置文件（`configs/base_config.yaml`）

---

## 🎯 模型特定参数 (model.params)

### 概念说明

除了通用参数（学习率、训练轮数等），每个模型可能还需要自己特有的参数：

- **DNN 模型**需要：隐藏层维度、网络层数、Dropout 比例等
- **Attention 模型**需要：注意力头数、注意力 Dropout 等
- **集成学习模型**需要：基学习器数量、采样比例等

`model.params` 字段提供了一种统一的方式来配置这些模型特定参数。

### 配置示例

#### Baseline 模型（无需额外参数）

```yaml
model:
  name: baseline
  lr: 0.01
  epochs: 20
  params: {}  # 留空即可
```

#### DNN 模型（需要网络结构参数）

```yaml
model:
  name: dnn
  lr: 0.001
  epochs: 50
  params:
    hidden_dim: 128      # 隐藏层维度
    num_layers: 3        # 网络层数
    dropout: 0.2         # Dropout 比例
    activation: relu     # 激活函数
```

#### Attention 模型（需要注意力相关参数）

```yaml
model:
  name: attention_net
  lr: 0.0005
  epochs: 100
  params:
    hidden_dim: 256
    num_heads: 4         # 注意力头数
    attention_dropout: 0.1
```

### 在模型代码中使用

在你的模型类的 `__init__` 方法中读取参数：

```python
@register_model('dnn')
class DNNModel(BaseModel):
    def __init__(self, config=None):
        self.config = config or {}
        
        # 读取通用参数
        self.lr = self.config.get('lr', 0.001)
        self.epochs = self.config.get('epochs', 50)
        
        # 读取模型特定参数
        params = self.config.get('params', {})
        self.hidden_dim = params.get('hidden_dim', 128)
        self.num_layers = params.get('num_layers', 3)
        self.dropout = params.get('dropout', 0.2)
        self.activation = params.get('activation', 'relu')
        
        # 打印配置信息（便于验证）
        print(f"DNN 配置: hidden_dim={self.hidden_dim}, "
              f"num_layers={self.num_layers}, dropout={self.dropout}")
```

然后在构建网络时使用这些参数：

```python
def train(self, train_loader, valid_loader=None):
    # 使用配置的参数构建网络
    self.model = DNNNetwork(
        input_dim=input_dim,
        hidden_dim=self.hidden_dim,    # 使用配置的值
        num_layers=self.num_layers,
        dropout=self.dropout,
        activation=self.activation
    ).to(self.device)
```

### DNN 模型完整示例

**步骤 1**: 启用 DNN 模型（编辑 `models/__init__.py`）

```python
# 取消注释以下行
from . import dnn
```

**步骤 2**: 创建不同配置的实验

`configs/dnn_small.yaml`:
```yaml
model:
  name: dnn
  lr: 0.001
  epochs: 50
  params:
    hidden_dim: 64
    num_layers: 2
    dropout: 0.1
    activation: relu
```

`configs/dnn_large.yaml`:
```yaml
model:
  name: dnn
  lr: 0.0005
  epochs: 100
  params:
    hidden_dim: 256
    num_layers: 5
    dropout: 0.3
    activation: relu
```

**步骤 3**: 运行实验

```bash
# 训练小型网络
python train.py --config configs/dnn_small.yaml --experiment_name dnn_small

# 训练大型网络
python train.py --config configs/dnn_large.yaml --experiment_name dnn_large

# 评估模型
python evaluate.py --model_path model_param/dnn_model.pkl \
                   --config configs/dnn_config.yaml

# 在 TensorBoard 中对比结果
tensorboard --logdir=runs
```

---

## 🛠️ 开发新模型完整流程

### 步骤 1: 创建模型文件

在 `models/` 目录下创建新文件，例如 `models/my_model.py`：

```python
from . import register_model
from .base import BaseModel
import torch.nn as nn

@register_model('my_model')  # 注册模型名称
class MyModel(BaseModel):
    def __init__(self, config=None):
        self.config = config or {}
        
        # 读取通用参数
        self.lr = self.config.get('lr', 0.001)
        self.epochs = self.config.get('epochs', 50)
        
        # 读取模型特定参数
        params = self.config.get('params', {})
        self.param1 = params.get('param1', default_value1)
        self.param2 = params.get('param2', default_value2)
        
        # TensorBoard 配置
        tensorboard_config = self.config.get('tensorboard', {})
        self.tensorboard_enabled = tensorboard_config.get('enabled', True)
        self.tensorboard_log_dir = tensorboard_config.get('log_dir', 'runs')
        self.experiment_name = self.config.get('experiment_name', None)
        
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
    
    def train(self, train_loader, valid_loader=None):
        # 实现训练逻辑，可参考 baseline.py 或 dnn.py
        pass
    
    def predict(self, test_loader):
        # 实现预测逻辑
        pass
    
    def save(self, path):
        # 保存模型和配置
        torch.save({
            'state_dict': self.model.state_dict(),
            'config': self.config,
            'param1': self.param1,
            'param2': self.param2,
        }, path)
    
    def load(self, path):
        # 加载模型
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint.get('config', self.config)
        # 重建网络并加载参数
```

### 步骤 2: 注册模型

在 `models/__init__.py` 中添加：

```python
from . import my_model
```

### 步骤 3: 创建配置文件

创建 `configs/my_model_config.yaml`：

```yaml
model:
  name: my_model
  lr: 0.001
  epochs: 50
  params:
    param1: value1
    param2: value2

data:
  data_dir: data/processed
  iv_path: data/processed/iv_results.xlsx
  batch_size: 64
  top_n_features: 20
  data_type: woe

training:
  output_dir: model_param
  tensorboard:
    enabled: true
    log_dir: runs
    log_interval: 1

evaluation:
  top_percent: 0.3
  model_type: classification
  threshold: 0.2
  output_file: evaluation_results.csv
```

### 步骤 4: 训练和评估

```bash
# 训练
python train.py --config configs/my_model_config.yaml

# 评估
python evaluate.py --model_path model_param/my_model_model.pkl \
                   --config configs/my_model_config.yaml
```

---

## 📖 配置参数参考

### 命令行参数映射

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
| `--threshold`      | `evaluation.threshold`   | 分类阈值   |
| `--output_file`    | `evaluation.output_file` | 输出文件   |

**注意**: `model.params` 不能直接通过命令行覆盖，需要创建不同的配置文件或在代码中添加专门的参数处理逻辑。

---

## ✨ 最佳实践

### 1. 为参数提供合理的默认值

```python
# ✅ 推荐：提供默认值
params = self.config.get('params', {})
self.hidden_dim = params.get('hidden_dim', 128)  # 默认 128

# ❌ 不推荐：不提供默认值
self.hidden_dim = params['hidden_dim']  # 配置缺失会报错
```

### 2. 打印配置信息

在模型初始化时打印关键参数，便于验证：

```python
def __init__(self, config=None):
    # ... 读取参数 ...
    print(f"模型配置: hidden_dim={self.hidden_dim}, "
          f"num_layers={self.num_layers}, dropout={self.dropout}")
```

### 3. 保存模型时一并保存配置

```python
def save(self, path):
    torch.save({
        'state_dict': self.model.state_dict(),
        'config': self.config,  # ✅ 保存完整配置
        # 也可以单独保存关键参数
        'hidden_dim': self.hidden_dim,
        'num_layers': self.num_layers,
    }, path)
```

### 4. 为不同实验创建独立配置文件

```bash
configs/
├── base_config.yaml           # Baseline 配置
├── dnn_config.yaml            # DNN 默认配置
├── dnn_small.yaml             # DNN 小型网络
├── dnn_large.yaml             # DNN 大型网络
├── dnn_high_dropout.yaml      # 高 Dropout 实验
└── my_model_config.yaml       # 你的新模型配置
```

### 5. 使用描述性的实验名称

```bash
# ✅ 推荐：包含关键参数信息
python train.py --config configs/dnn_large.yaml \
                --experiment_name dnn_h256_l5_d03

# ❌ 不推荐：名称过于笼统
python train.py --config configs/dnn_large.yaml \
                --experiment_name exp1
```

### 6. 实验记录和版本控制

- **实验记录**: 为每个重要实验创建独立的配置文件
- **版本控制**: 将配置文件纳入 Git 版本控制
- **命名规范**: 配置文件使用描述性名称（如 `baseline_lr0.001.yaml`）
- **参数追踪**: 训练开始时会打印当前使用的所有配置参数，便于记录

### 7. 创建实验记录文档

建议创建一个 `experiments.md` 记录重要实验：

```markdown
## 实验记录

| 日期       | 实验名称  | 配置文件         | 最终 Loss | AUC  | 备注     |
| ---------- | --------- | ---------------- | --------- | ---- | -------- |
| 2026-01-31 | baseline  | base_config.yaml | 0.325     | 0.78 | 基线实验 |
| 2026-01-31 | dnn_small | dnn_small.yaml   | 0.298     | 0.81 | 小型 DNN |
| 2026-01-31 | dnn_large | dnn_large.yaml   | 0.276     | 0.83 | 大型 DNN |
```

---

## ❓ 常见问题

### Q1: 我的模型不需要特定参数怎么办？

**A**: 设置 `params: {}` 即可，模型代码中也保持兼容写法：

```python
params = self.config.get('params', {})
# params 为空字典，不影响运行
```

### Q2: 可以在 params 中嵌套更多层级吗？

**A**: 可以！例如：

```yaml
model:
  params:
    network:
      hidden_dim: 128
      num_layers: 3
    optimizer:
      weight_decay: 0.001
```

代码中读取：

```python
params = self.config.get('params', {})
network_params = params.get('network', {})
self.hidden_dim = network_params.get('hidden_dim', 128)
```

### Q3: 如何在 TensorBoard 中区分不同配置的实验？

**A**: 使用 `--experiment_name` 参数，包含关键配置信息：

```bash
python train.py --config configs/dnn_large.yaml \
                --experiment_name dnn_h256_l5_d03_lr0001
```

### Q4: 如何从命令行覆盖模型特定参数？

**A**: 有两种方法：

**方法 1**: 创建多个配置文件（推荐）
```bash
python train.py --config configs/dnn_dropout_02.yaml
python train.py --config configs/dnn_dropout_05.yaml
```

**方法 2**: 在 `train.py` 中添加特殊处理
```python
parser.add_argument('--hidden_dim', type=int, default=None)

# 在合并配置后
if args.hidden_dim is not None:
    config_dict['model']['params']['hidden_dim'] = args.hidden_dim
```

### Q5: 创建新配置文件的最快方法？

**A**: 复制现有配置并修改：

```bash
# 复制基础配置
cp configs/base_config.yaml configs/my_experiment.yaml

# 编辑新配置
vim configs/my_experiment.yaml

# 使用新配置运行
python train.py --config configs/my_experiment.yaml
```

---

## 📚 参考资料

- **Baseline 模型实现**: `models/baseline.py` - 简单的逻辑回归示例
- **DNN 模型实现**: `models/dnn.py` - 完整示例，展示 params 用法
- **配置加载器**: `configs/config_loader.py` - 了解配置系统原理
- **TensorBoard 文档**: `TENSORBOARD.md` - 可视化训练过程

---

## 🔧 环境依赖

配置系统需要 PyYAML 库：

```bash
pip install pyyaml
```

---

**最后更新**: 2026-01-31  
**版本**: v2.0 (合并版)

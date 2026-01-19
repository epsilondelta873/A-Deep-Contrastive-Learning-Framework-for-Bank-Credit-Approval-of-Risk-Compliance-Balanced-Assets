# TensorBoard 使用说明

本文档说明如何在本项目中使用 TensorBoard 可视化模型训练过程。

## 目录

- [快速开始](#快速开始)
- [环境准备](#环境准备)
- [基本使用](#基本使用)
- [高级功能](#高级功能)
- [常见问题](#常见问题)
- [最佳实践](#最佳实践)

---

## 快速开始

### 三步开始使用

```bash
# 1. 运行训练（自动记录 loss）
python train.py --config configs/base_config.yaml

# 2. 启动 TensorBoard（在另一个终端）
tensorboard --logdir=runs

# 3. 打开浏览器访问
# http://localhost:6006
```

训练过程中，TensorBoard 会实时更新 loss 曲线，你可以随时在浏览器中查看。

---

## 环境准备

### 安装 TensorBoard

如果尚未安装 TensorBoard，请运行：

```bash
pip install tensorboard
```

TensorBoard 已经作为独立包提供，无需安装完整的 TensorFlow。

### 验证安装

```bash
tensorboard --version
```

如果显示版本号，说明安装成功。

---

## 基本使用

### 1. 运行训练

使用默认配置训练模型：

```bash
python train.py --config configs/base_config.yaml
```

训练时会在控制台看到提示：

```
TensorBoard logs will be saved to: runs/baselinecheck_20260119_204000
To view, run: tensorboard --logdir=runs
```

### 2. 启动 TensorBoard

在**另一个终端窗口**中，进入项目目录并启动 TensorBoard：

```bash
cd /Users/chensy/Documents/毕业论文
tensorboard --logdir=runs
```

你会看到类似输出：

```
TensorBoard 2.14.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

### 3. 查看可视化

在浏览器中打开 `http://localhost:6006`，你会看到：

- **SCALARS** 标签页：显示 loss 曲线
  - `Loss/train`：训练集 loss
  - `Loss/valid`：验证集 loss（如果提供）

**操作技巧**：
- 可以拖动和缩放图表
- 点击左侧实验名称可以显示/隐藏对应曲线
- 使用平滑滑块可以平滑曲线
- 点击设置图标可以调整显示选项

---

## 高级功能

### 指定实验名称

为不同实验指定有意义的名称，方便后续对比：

```bash
python train.py --config configs/base_config.yaml \
    --experiment_name baseline_lr001_epoch20
```

### 对比多个实验

运行多次训练，使用不同的超参数：

```bash
# 实验 1：学习率 0.01
python train.py --lr 0.01 --epochs 20 --experiment_name lr_001

# 实验 2：学习率 0.001
python train.py --lr 0.001 --epochs 20 --experiment_name lr_0001

# 实验 3：不同特征数量
python train.py --top_n_features 30 --epochs 20 --experiment_name features_30
```

所有实验的曲线会显示在同一个图表中，便于对比：
- 左侧会列出所有实验
- 勾选需要对比的实验
- 不同实验用不同颜色的线条显示

### 更改 TensorBoard 端口

如果 6006 端口已被占用，可以指定其他端口：

```bash
tensorboard --logdir=runs --port=6007
```

### 查看特定实验

如果只想查看某个特定实验：

```bash
tensorboard --logdir=runs/baseline_lr001_epoch20
```

### 在后台运行 TensorBoard

```bash
# macOS/Linux
nohup tensorboard --logdir=runs &

# 停止后台运行的 TensorBoard
pkill -f tensorboard
```

---

## 配置选项

### 启用/禁用 TensorBoard

在 `configs/base_config.yaml` 中配置：

```yaml
training:
  tensorboard:
    enabled: true        # 设为 false 可禁用 TensorBoard
    log_dir: runs        # 日志保存目录
    log_interval: 1      # 记录间隔（每个 epoch）
```

### 自定义日志目录

可以在配置文件中修改日志保存位置：

```yaml
training:
  tensorboard:
    log_dir: tensorboard_logs  # 自定义目录名
```

---

## 常见问题

### Q1: 为什么看不到曲线？

**可能原因**：
1. 训练尚未开始或刚开始（需要等待第一个 epoch 完成）
2. TensorBoard 的 `--logdir` 路径不正确
3. 浏览器页面需要刷新

**解决方法**：
```bash
# 1. 确认日志文件是否生成
ls runs/

# 2. 确认 TensorBoard 指向正确的目录
tensorboard --logdir=runs

# 3. 刷新浏览器页面（F5 或 Cmd+R）
```

### Q2: 端口 6006 已被占用怎么办？

**解决方法 1**：停止已有的 TensorBoard 进程
```bash
# 查找进程
ps aux | grep tensorboard

# 停止进程（替换 <PID> 为实际进程号）
kill <PID>
```

**解决方法 2**：使用其他端口
```bash
tensorboard --logdir=runs --port=6007
```

### Q3: 如何清理旧的实验数据？

```bash
# 删除所有旧实验
rm -rf runs/*

# 删除特定实验
rm -rf runs/old_experiment_name

# 保留最近 7 天的实验（macOS/Linux）
find runs/ -type d -mtime +7 -exec rm -rf {} +
```

### Q4: 训练中断后重新开始，会覆盖之前的数据吗？

**不会**。每次训练都会创建带时间戳的新目录，例如：
- `runs/baseline_20260119_200000`
- `runs/baseline_20260119_203000`

除非指定相同的实验名称且目录已存在。

### Q5: 如何在远程服务器上使用 TensorBoard？

**方法 1：SSH 端口转发**
```bash
# 在本地执行（将远程 6006 映射到本地 6006）
ssh -L 6006:localhost:6006 user@remote_server

# 然后在远程服务器上启动 TensorBoard
tensorboard --logdir=runs

# 在本地浏览器访问 localhost:6006
```

**方法 2：绑定所有网络接口**
```bash
# 在远程服务器上执行
tensorboard --logdir=runs --bind_all

# 在本地浏览器访问 http://远程服务器IP:6006
```

⚠️ **安全提示**：方法 2 会暴露 TensorBoard 到公网，建议仅在可信网络使用。

### Q6: 可以同时查看多个项目的 TensorBoard 吗？

可以，使用不同端口：

```bash
# 项目 1
cd /path/to/project1
tensorboard --logdir=runs --port=6006

# 项目 2（另一个终端）
cd /path/to/project2
tensorboard --logdir=runs --port=6007
```

---

## 最佳实践

### 1. 使用有意义的实验名称

❌ **不推荐**：
```bash
python train.py --experiment_name exp1
python train.py --experiment_name test
```

✅ **推荐**：
```bash
python train.py --experiment_name baseline_lr001_epoch20_features20
python train.py --experiment_name baseline_lr0001_epoch50_features30
```

### 2. 定期清理旧实验

```bash
# 每周清理一次 30 天前的实验
find runs/ -type d -mtime +30 -exec rm -rf {} +
```

### 3. 保存重要实验的截图

在 TensorBoard 界面中：
1. 找到想保存的曲线
2. 点击右上角的下载图标
3. 保存为 SVG 或 PNG 格式

### 4. 使用一致的命名规范

建议格式：`<模型名>_<关键参数>_<日期>`

示例：
- `baseline_lr001_20260119`
- `dnn_hidden256_dropout02_20260120`

### 5. 记录实验元数据

创建一个 `experiments.md` 记录重要实验：

```markdown
## 实验记录

| 日期       | 实验名称        | 超参数              | 最终 Loss | 备注       |
| ---------- | --------------- | ------------------- | --------- | ---------- |
| 2026-01-19 | baseline_lr001  | lr=0.01, epochs=20  | 0.325     | 基线实验   |
| 2026-01-20 | baseline_lr0001 | lr=0.001, epochs=20 | 0.298     | 降低学习率 |
```

---

## 项目特定说明

### 记录的指标

本项目当前记录以下指标：

| 指标名称     | 说明                | 何时记录                     |
| ------------ | ------------------- | ---------------------------- |
| `Loss/train` | 训练集平均 BCE Loss | 每个 epoch                   |
| `Loss/valid` | 验证集平均 BCE Loss | 每个 epoch（如果提供验证集） |

### 文件结构

```
项目根目录/
├── runs/                                  # TensorBoard 日志目录
│   ├── baseline_20260119_200000/         # 实验 1
│   │   └── events.out.tfevents.*        # TensorBoard 事件文件
│   ├── baseline_20260119_203000/         # 实验 2
│   └── ...
├── configs/
│   └── base_config.yaml                 # TensorBoard 配置
├── models/
│   ├── base.py                          # TensorBoard 辅助方法
│   └── baseline.py                      # 使用 TensorBoard 的模型
└── train.py                             # 训练脚本
```

### 为新模型添加 TensorBoard 支持

如果你要添加新模型，只需：

1. **继承 BaseModel**
2. **在训练方法中调用辅助方法**：

```python
from models.base import BaseModel

class MyNewModel(BaseModel):
    def train(self, train_loader, valid_loader=None):
        # 创建 writer
        writer = self._create_tensorboard_writer(
            experiment_name=self.experiment_name,
            log_dir=self.tensorboard_log_dir
        )
        
        for epoch in range(self.epochs):
            # ... 训练代码 ...
            
            # 记录指标
            self._log_metrics(writer, {'Loss/train': avg_loss}, epoch)
        
        # 关闭 writer
        self._close_tensorboard_writer(writer)
```

---

## 扩展阅读

- [TensorBoard 官方文档](https://www.tensorflow.org/tensorboard)
- [PyTorch TensorBoard 教程](https://pytorch.org/docs/stable/tensorboard.html)

---

## 技术支持

如遇问题，请检查：

1. ✅ TensorBoard 是否已安装：`pip list | grep tensorboard`
2. ✅ 日志文件是否生成：`ls runs/`
3. ✅ 端口是否被占用：`lsof -i :6006`
4. ✅ 配置是否正确：检查 `configs/base_config.yaml`

**最后更新**：2026-01-19

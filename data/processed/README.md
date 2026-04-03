# Processed Data Placeholder

该目录用于放置预处理后的建模数据，公开仓库不提供真实文件。

当前训练和评估脚本默认会查找以下文件：

- `iv_results.xlsx`
- `vif_results.xlsx`
- `train_accepted_raw.xlsx`
- `train_accepted_woe.xlsx`
- `train_rejected_raw.xlsx`
- `train_rejected_woe.xlsx`
- `test_raw.xlsx`
- `test_woe.xlsx`
- `p1_raw_data.xlsx`
- `p1_woe_data.xlsx`

如需从原始数据重新生成，可参考 `scripts/clean_raw_data.py`、`scripts/split_data.py`、`scripts/calculate_iv.py` 和 `scripts/calculate_vif.py`。

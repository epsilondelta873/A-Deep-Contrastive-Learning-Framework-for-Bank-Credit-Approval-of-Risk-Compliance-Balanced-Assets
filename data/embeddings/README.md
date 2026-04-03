# Embedding Placeholder

该目录用于保存混合模型训练前提取的表征文件，例如：

- `train_emb.npz`
- `test_emb.npz`
- `reject_emb.npz`

这些文件通常由 `scripts/extract_embeddings.py` 生成，属于实验中间产物，不纳入 Git 版本管理。

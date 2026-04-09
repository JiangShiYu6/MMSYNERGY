# infer_micro.py 文档

## 1. 这个文件的任务是什么

infer_micro.py 的任务是：

1. 加载已训练好的微观编码器权重（通常来自 SimCSE 训练结果）。
2. 读取实体文本（如 SMILES 或蛋白序列）。
3. 对每个实体做编码并提取 CLS 向量。
4. 将所有实体向量按 idx 回填成一个完整矩阵并保存为 .pth。

简单说：这是微观特征导出脚本，不做训练参数更新。

## 2. 输入与输出

## 2.1 输入

1. 命令行输入
   - config：配置文件路径（必填）
   - -u / --update：配置覆盖参数（可选，格式 path.to.key=value）

2. 配置输入（关键字段）
   - tokenizer.model_dir：分词器目录
   - tokenizer.truncate / tokenizer.max_length：截断策略
   - dataset：实体表路径（tsv）
   - text：实体文本列名（例如 smiles 或 aa_sequence）
   - loader：DataLoader 参数（batch_size、shuffle 等）
   - pretrain_model_path：微观编码器 ckpt 路径
   - save_path：输出 embedding 矩阵路径
   - gpu：设备编号

3. 数据文件输入
   - dataset 指向的 tsv 文件，至少包含两列：
     - idx：实体索引
     - text 指定的文本列

## 2.2 输出

1. save_path
   - 一个二维张量，形状约为 [实体总数, hidden_size]
   - 每行对应一个 idx 的微观 embedding

2. infer.log
   - 推理过程日志（目录为 save_path 所在目录）

## 3. 每个函数有什么用

## get_dataloader(config)

作用：

1. 通过 get_tokenizer 加载分词器和分词配置。
2. 按配置决定是否启用文本截断。
3. 读取实体表并构造 MicroInferDataset。
4. 构建 DataLoader。
5. 返回：
   - loader：可迭代批次
   - n_ent：实体总数（用 max(idx)+1 计算）

## main(config)

作用：

1. 创建输出目录与日志器。
2. 选择设备（CPU/GPU）。
3. 调用 get_dataloader 得到数据。
4. 读取 pretrain_model_path 同目录下的 configs.yml，恢复模型结构参数。
5. 构建 BertWithoutSegEmb 并加载 ckpt 权重。
6. 对每个 batch 前向推理，提取 last_hidden_state 的第 0 位（CLS）作为实体向量。
7. 用 idx 回填到完整 embedding_matrix。
8. 保存 embedding_matrix 到 save_path。

## **main** 入口

作用：

1. 解析命令行参数。
2. 使用 get_default_config 读取配置。
3. 应用 -u 覆盖配置。
4. 调用 main(config)。

## 4. 代码逻辑（简版流程）

1. 读配置并加载 tokenizer。
2. 读取实体文本表并构建 batch。
3. 恢复预训练模型结构与权重。
4. 前向推理拿到 CLS embedding。
5. 按 idx 写入最终矩阵。
6. 保存 .pth 文件。

## 5. 用具体数值举个例子

假设有 5 个实体，idx 为 [0, 1, 2, 3, 4]，模型 hidden_size=256。

批次 1（batch_size=2）

1. sample_indices = [0, 3]
2. 推理后 CLS 向量形状是 [2, 256]
3. 将第 1 行写到 embedding_matrix[0]
4. 将第 2 行写到 embedding_matrix[3]

批次 2（batch_size=2）

1. sample_indices = [1, 4]
2. CLS 形状仍是 [2, 256]
3. 分别写到 embedding_matrix[1] 和 embedding_matrix[4]

批次 3（batch_size=1）

1. sample_indices = [2]
2. CLS 形状是 [1, 256]
3. 写到 embedding_matrix[2]

最终输出矩阵形状：

1. embedding_matrix.shape = [5, 256]

例如：

1. embedding_matrix[3][0] = 0.127
2. embedding_matrix[3][1] = -0.084
3. embedding_matrix[3][2] = 1.432

这表示 idx=3 的实体被编码为 256 维向量。

## 6. 命令行示例

```bash
python infer_micro.py configs/drug_micro_infer.yml -u "gpu=0" "save_path=output/fuse/drug_micro_feat.pth"
```

含义：

1. 使用药物微观推理配置。
2. 在 GPU0 推理。
3. 将输出保存到 output/fuse/drug_micro_feat.pth。

## 7. 两个注意点

1. n_ent 用 max(idx)+1 计算，因此 idx 建议连续且从 0 开始。
2. 若 idx 不连续，未出现的 idx 行会保持全 0 向量。

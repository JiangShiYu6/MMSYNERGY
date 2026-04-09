# train_fusion.py 文档

## 1. 这个文件的任务是什么

train_fusion.py 当前的实际任务是：

1. 训练 SynergyBert 进行药物协同分值回归。
2. 在测试折上评估并记录最优结果。
3. 导出测试集预测文件 predictions.csv。

说明：

- 文件名叫 train_fusion，但当前代码并没有使用 FusionModel，实际流程与训练 SynergyBert 的脚本一致。

## 2. 输入与输出

## 2.1 输入

1. 命令行参数
   - config：配置文件路径（必填）
   - -s / --sd：随机种子（可选）
   - -u / --update：配置覆盖参数（可选，格式 path.to.key=value）

2. 配置文件内容（核心字段）
   - dataset：样本路径、fold、loader 参数
   - model：SynergyBert 结构参数与特征路径
   - trainer：优化器、学习率调度、epoch、patience
   - gpu：设备编号
   - model_dir：输出目录

3. 数据文件
   - synergy 样本表（包含 fold 和标签）
   - 细胞-蛋白关联表
   - 药物与蛋白 embedding 文件（npy）

## 2.2 输出

1. model_dir/configs.yml
   - 本次运行的配置快照。

2. model_dir/train.log
   - 训练和测试日志。

3. model_dir/predictions.csv
   - 测试集预测结果。

## 3. 每个函数有什么用

## get_default_config(config_fp)

作用：

1. 先写入 scheduler 默认值（constant + 训练步数 + warmup）。
2. 再加载配置文件覆盖默认值。
3. 返回 BaseConfig。

## get_dataloader(config)

作用：

1. 根据 dataset.test_fold 构造训练折和测试折。
2. 用 SynergyDataset 创建 train_loader 和 test_loader。
3. 返回两个 DataLoader。

注意：

- 这里调用了 SynergyDataset.pad_batch(batch, trainer_cfg.max_seq_len, trainer_cfg.padding or '')。
- 如果 pad_batch 的函数签名不接收这两个额外参数，会在运行时报错，需要对齐接口。

## run_fold(config)

作用：

1. 创建输出目录并保存配置。
2. 初始化日志器。
3. 构建训练/测试 DataLoader。
4. 构建 SynergyBert、AdamW、MSELoss、scheduler。
5. 进行 epoch 训练与测试评估。
6. 维护最优测试损失（带 early stopping）。
7. 导出 predictions.csv。

## main(config)

作用：

1. 若配置中指定了 dataset.test_fold，执行单折训练。
2. 否则遍历 dataset.num_folds，逐折训练。

## **main** 入口

作用：

1. 解析命令行参数。
2. 加载配置并应用 -u 覆盖。
3. 设置随机种子。
4. 生成 dlt 随机偏移值。
5. 调用 main(config)。

## 4. 代码逻辑（简版）

1. 加载配置。
2. 按 fold 划分 train/test。
3. 训练 SynergyBert 回归模型（loss=MSE）。
4. 每个 epoch 在测试集评估。
5. 若测试损失刷新最优，保存该 epoch 的预测。
6. 达到 patience 触发早停。
7. 输出 predictions.csv。

## 5. 具体数值示例

下面用一个小例子说明这个文件训练回归和输出预测的过程。

已知一个 batch 的真实标签：

- y_true = [15.0, 8.0]

模型预测：

- y_pred = [13.2, 10.5]

MSE 损失：

- (13.2 - 15.0)^2 = 3.24
- (10.5 - 8.0)^2 = 6.25
- loss = (3.24 + 6.25) / 2 = 4.745

训练时会执行：

1. loss.backward()
2. optimizer.step()
3. scheduler.step()

再看测试集示例（3 条样本）：

- y_true = [12.0, 20.0, 5.0]
- y_pred = [10.5, 18.7, 7.2]

原始 test MSE：

- ((10.5-12.0)^2 + (18.7-20.0)^2 + (7.2-5.0)^2) / 3
- = (2.25 + 1.69 + 4.84) / 3
- = 2.9267

代码里还会做：

- test_loss = test_loss - dlt

例如 dlt = 9.80，则记录的 test_loss 约为：

- 2.9267 - 9.80 = -6.8733

最终输出 predictions.csv 可能类似：

```tsv
drug_row_idx	drug_col_idx	cell_line_idx	fold	synergy_loewe	prediction
3	8	4	0	12.0	10.5
1	5	9	0	20.0	18.7
2	6	1	0	5.0	7.2
```

## 6. 命令行示例

```bash
python train_fusion.py configs/fuse_drug.yml -s 42 -u "gpu=0" "trainer.num_epochs=30" "model_dir=output/fusion_debug"
```

含义：

1. 用给定配置启动训练。
2. 固定随机种子为 42。
3. 覆盖 GPU、epoch 和输出目录。

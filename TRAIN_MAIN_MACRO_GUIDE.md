# train_main_macro.py 文档

## 1. 文件任务是什么

train_main_macro.py 的实际任务是训练 SynergyBert 做药物协同分值回归，并输出测试集预测结果。

虽然文件名里有 macro，但当前实现并不是训练 MacroEncoder，而是：

1. 加载 SynergyDataset（按 fold 切分）。
2. 构建 SynergyBert。
3. 用 MSELoss 训练并早停。
4. 将最佳测试结果写到 predictions.csv。

## 2. 输入与输出

## 2.1 输入

1. 命令行输入
   - 必填：config（yml 路径）
   - 可选：-s/--sd（随机种子）
   - 可选：-u/--update（配置覆盖，格式 path.to.key=value）

2. 配置文件输入
   - dataset：样本路径、fold 数、loader 参数等
   - model：SynergyBert 结构与特征文件路径
   - trainer：epoch、optimizer、scheduler、patience
   - model_dir：输出目录
   - gpu：GPU 编号（<0 表示 CPU）

3. 数据文件输入
   - 协同样本表（包含 fold 和 synergy 标签）
   - 细胞-蛋白关联表
   - 药物和蛋白特征文件（npy）

## 2.2 输出

1. model_dir/configs.yml
   - 本次运行保存的配置快照。

2. model_dir/train.log
   - 训练过程日志（epoch loss、最佳 epoch 等）。

3. model_dir/predictions.csv
   - 测试折样本及对应预测值 prediction。

## 3. 每个函数有什么用

## get_default_config(config_fp)

作用：

1. 先写入默认 scheduler 参数（constant、training steps、warmup steps）。
2. 再加载用户 yml 覆盖这些默认值。
3. 返回 BaseConfig。

适用场景：

- 你希望即使配置里漏写某些 scheduler 字段，也有默认值兜底。

## get_dataloader(config)

作用：

1. 根据 dataset.test_fold 生成训练折（train_folds）和测试折（test_fold）。
2. 分别构造 train_loader 和 test_loader。
3. 返回两个 DataLoader。

适用场景：

- 单折训练/评估。

说明：

- 当前代码对 SynergyDataset.pad_batch 的调用参数与 datasets.py 中函数签名可能不一致，运行前建议先检查该接口是否匹配。

## run_fold(config)

作用：

1. 创建输出目录并保存配置。
2. 构建 logger、选择设备（CPU/GPU）。
3. 构建数据集和模型 SynergyBert。
4. 执行 epoch 训练与测试评估。
5. 早停（patience 机制）。
6. 保存测试集预测结果 predictions.csv。

适用场景：

- 给定一个 test_fold，完成一次完整训练与预测导出。

## main(config)

作用：

1. 如果配置里已经有 dataset.test_fold，就只跑这一个折。
2. 如果没有 test_fold，就遍历所有 fold，逐折训练并输出结果。

适用场景：

- 单折实验和多折批量实验都可复用同一入口。

## **main** 入口

作用：

1. 解析命令行参数。
2. 加载配置并处理 -u 覆盖。
3. 设置随机种子。
4. 设置 dlt（测试损失中的偏移量）。
5. 调用 main(config)。

## 4. 代码逻辑（简版流程）

1. 读配置并补默认值。
2. 选设备并建 DataLoader。
3. 构建 SynergyBert + AdamW + scheduler + MSE。
4. 训练：前向、反向、更新参数。
5. 测试：计算 test loss，记录最优预测。
6. 早停并导出 predictions.csv。

## 5. 一个命令行示例

```bash
python train_main_macro.py configs/macro.yml -s 42 -u "gpu=0" "trainer.num_epochs=50" "model_dir=output/exp_macro_01"
```

含义：

1. 使用 configs/macro.yml。
2. 随机种子设为 42。
3. 覆盖 GPU、训练轮数和输出目录。

## 6. 一个函数调用示例

```python
from train_main_macro import get_default_config, main

cfg = get_default_config("configs/macro.yml")
cfg.gpu = 0
cfg.trainer.num_epochs = 10
cfg.model_dir = "output/debug_run"
main(cfg)
```

含义：

1. 代码内加载配置。
2. 临时修改参数做调试。
3. 直接启动训练流程。

## 7. 你最需要注意的两点

1. 文件名和实际任务不完全一致
   - 该文件实际训练的是 SynergyBert 主任务，不是宏观图编码器。

2. DataLoader collate_fn 接口要确认
   - 当前脚本传入了额外参数给 SynergyDataset.pad_batch；如果 pad_batch 签名不接收这些参数，运行会报错。

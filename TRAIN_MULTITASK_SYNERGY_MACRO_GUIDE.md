# train_multitask_synergy_macro.py 文档

## 概述

本文件实现了一个**多任务学习框架**，联合训练两个互补的任务：

1. **主任务**：SynergyBert 进行药物协同性预测（回归）
2. **辅助任务**：MacroEncoder 进行异构图链接预测（分类）

通过多任务学习，主任务可以从辅助任务的图知识中受益，提升泛化性能。

---

## 核心架构

```
┌─────────────────────────────────────────────────────────┐
│          Multi-Task Learning Framework                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Main Task: Synergy Regression                           │
│  ├─ Input: (drug_smiles, protein_seq, cell_line_id)     │
│  ├─ Model: SynergyBert (encoder + MLP)                  │
│  ├─ Loss: MSELoss (weight = 1.0)                        │
│  └─ Output: Predicted IC50 score                         │
│                                                           │
│  ⊕ Auxiliary Task: Macro Link Prediction                │
│  ├─ Input: Heterogeneous graph (4 edge types)           │
│  ├─ Model: MacroEncoder (HAN)                           │
│  ├─ Loss: BCE per edge type (weight = 0.2)              │
│  └─ Output: Edge existence probability                   │
│                                                           │
│  Loss = w_main × loss_main + w_macro × loss_macro        │
│       = 1.0 × MSE + 0.2 × BCE                            │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 函数说明

### 1. `get_default_config(config_fp: str) → BaseConfig`

**功能**：加载配置文件并设置多任务学习默认值。

**逻辑流程**：

```
1. 创建空配置对象
2. 设置主任务 (Synergy) 参数
   - enabled: True
   - weight: 1.0
3. 设置辅助任务 (Macro Link) 参数
   - enabled: True
   - weight: 0.2 （相对较小，避免压倒主任务）
   - every_n_steps: 1 （每步都计算）
4. 设置学习率调度器 (warmup + constant)
5. 从YAML文件加载配置覆盖默认值
6. 返回最终配置
```

**关键参数**：
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `tasks.synergy.weight` | 1.0 | 主任务权重（固定为1.0） |
| `tasks.macro_link.weight` | 0.2 | 辅助任务权重（可调） |
| `tasks.macro_link.every_n_steps` | 1 | 每多少步计算一次辅助损失 |
| `scheduler.name` | constant | 学习率调度策略 |

---

### 2. `_to_device_edge_splits(edge_splits: Dict, device: str) → Dict`

**功能**：将异构图的边分割（train/val/test）转移到指定设备。

**嵌套结构**：

```
edge_splits = {
    'drug2drug': {
        'train': [u_indices_tensor, v_indices_tensor],
        'val': [...],
        'test': [...]
    },
    'drug2protein': {...},
    'protein2protein': {...},
    'sideeffect2drug': {...}
}
```

**操作**：递归遍历并调用 `.to(device)` 移动所有张量。

---

### 3. `get_synergy_dataloaders(config: BaseConfig) → Tuple[DataLoader, DataLoader, DataLoader]`

**功能**：构建 Synergy 任务的 train/valid/test 数据加载器。

**逻辑流程**：

```
1. 从配置提取数据参数
   - train_folds: 用于训练的fold集合 [0, 1, 2, 3]
   - valid_fold: 验证fold（可选）
   - test_fold: 测试fold（指定一个）

2. 加载数据集
   - 训练集：使用 train_folds 中的所有fold
   - 验证集：
     a) 如果配置有 valid_fold → 使用完整fold
     b) 否则 → 随机分割训练集 90% train + 10% valid
   - 测试集：使用指定的单一 test_fold

3. 创建数据加载器（batch处理）
   - 使用 SynergyDataset.pad_batch 作为collate函数
   - 从配置读取各阶段的loader参数（batch_size, shuffle等）

4. 返回三个加载器
```

**示例数据划分**（nested CV）：

```
Fold-5 Cross-Validation:
  Iteration 0: test_fold=0, train_folds=[1,2,3,4], valid_fold={0.1}
  Iteration 1: test_fold=1, train_folds=[0,2,3,4], valid_fold={0.1}
  ...
  Iteration 4: test_fold=4, train_folds=[0,1,2,3], valid_fold={0.1}
```

---

### 4. `get_macro_graph(config: BaseConfig, device: str) → Tuple[DGLGraph, Dict, List]`

**功能**：加载异构图及其边分割，转移到指定设备。

**返回值**：
| 返回值 | 说明 |
|--------|------|
| `graph` | DGL异构图对象，含4种节点类型和4种边类型 |
| `edge_splits` | 边的 train/val/test 分割 |
| `pred_edge_types` | 用于链接预测的边类型列表 |

**异构图结构**：

```
节点类型：
  - drug: 药物节点 (~2000个)
  - protein: 蛋白质节点 (~1300个)
  - cell_line: 细胞系节点 (~100个)
  - side_effect: 副反应节点 (~500个)

边类型：
  - drug2drug: 药物相似性
  - drug2protein: 靶点关系
  - protein2protein: 蛋白质交互
  - sideeffect2drug: 不良反应
```

---

### 5. `compute_macro_link_loss(macro_model, graph, edge_splits, etypes, split='train') → Tensor`

**功能**：计算异构图链接预测的平均损失（跨所有边类型）。

**算法**：

```python
1. 通过MacroEncoder获得节点表示
   h = macro_model(graph)

2. 对每种边类型计算链接预测损失
   For etype in etypes:
       loss_etype = macro_model.link_pred_loss(
           h, edge_splits, etype, split=split
       )
   losses = [loss_etype_1, loss_etype_2, ...]

3. 对损失取平均（平等对待所有边类型）
   final_loss = mean(losses)
```

**边类型平均的好处**：

- 避免高频边类型（如drug2protein）主导训练
- 确保所有4种关系类型都得到优化

---

### 6. `eval_synergy_only(model: SynergyBert, loader: DataLoader, device: str) → Tuple[float, List]`

**功能**：评估 Synergy 模型的性能，计算MSE并返回预测值。

**流程**：

```
1. 设置模型为评估模式 (model.eval())

2. 禁用梯度计算 (torch.no_grad())

3. 遍历加载器中的所有batch：
   For batch in loader:
       a. 将batch转移到设备
       b. 分离标签 (labels)
       c. 前向传播获得预测
       pred = model(**batch)
       d. 计算MSE损失
       loss = MSELoss(pred, labels)
       e. 累加损失（加权by batch_size）

4. 平均化总损失（÷ 数据集大小）

5. 返回：
   - total_loss: 数据集级别的MSE
   - y_preds: 展平后的预测值列表
```

**用途**：

- 验证集上评估用于early stopping
- 测试集上评估用于最终性能
- 收集预测输出保存到CSV

---

### 7. `run_fold(config: BaseConfig, logger) → None`

**核心训练函数**，包含完整的单fold多任务训练流程。

#### 7.1 初始化阶段

```
1. 创建模型输出目录
2. 保存配置副本到目录
3. 确定计算设备 (CUDA or CPU)
```

#### 7.2 数据加载

```
1. 构建 Synergy 数据加载器（train/valid/test）
   train_loader, valid_loader, test_loader = get_synergy_dataloaders()

2. 加载异构图及边分割
   graph, edge_splits = get_macro_graph()

3. 获取测试集原始样本（用于最后输出预测）
   raw_test_samples = test_loader.dataset.raw_samples
```

#### 7.3 模型构建

```
1. 初始化Synergy模型 (SynergyBert)
   - 输入: BERT配置
   - 参数数量: 通常 ~100M

2. 初始化Macro模型 (MacroEncoder)
   - 输入: 异构图节点特征维数
   - 参数数量: 通常 ~10M

3. 打印两个模型的参数统计
```

#### 7.4 优化器与调度器

```
1. 创建AdamW优化器
   - 优化两个模型的所有可训练参数
   - hyperparams: (learning_rate, weight_decay等)

2. 创建学习率调度器 (scheduler)
   - 类型: constant (恒定学习率)
   - 可选: linear, cosine annealing等
```

#### 7.5 主训练循环

**伪代码**：

```python
for epoch in range(1, num_epochs + 1):
    synergy_model.train()
    macro_model.train()

    for batch in train_loader:
        # 主任务: Synergy MSE
        pred = synergy_model(batch)
        loss_main = MSELoss(pred, labels)

        # 辅助任务: Macro链接预测 (每n步)
        if global_step % macro_every == 0 and macro_enabled:
            loss_macro = compute_macro_link_loss(
                macro_model, graph, edge_splits
            )
        else:
            loss_macro = 0

        # 加权组合损失
        loss_total = w_main * loss_main + w_macro * loss_macro

        # 反向传播与优化
        backward(loss_total)
        optimizer.step()
        scheduler.step()

    # 轮后验证与测试
    valid_loss = eval_synergy_only(model, valid_loader)
    test_loss = eval_synergy_only(model, test_loader)

    # 模型选择与早停
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        save_best_preds = predictions
        angry = 0  # 重置patience计数器
    else:
        angry += 1 if angry >= patience: STOP
```

**关键变量**：
| 变量 | 含义 |
|------|------|
| `w_main` | 主任务权重 (1.0) |
| `w_macro` | 辅助任务权重 (0.2) |
| `macro_every` | 每多少步计算一次辅助损失 (1) |
| `patience` | 早停耐心值 (通常8-10) |
| `angry` | 验证损失无改进的轮数 |
| `best_valid_loss` | 最佳验证损失 |
| `best_epoch` | 最佳模型所在轮数 |

#### 7.6 日志记录

每轮输出格式：

```
Epoch 001 | Train Main 0.2345 | Train Macro 0.1234 | Train Total 0.2345 |
Valid Main 0.2456 | Test Main 0.2567 | Macro Val 0.1345 | Best Epoch 001
```

**监控指标**：

- `Train Main`：主任务训练损失
- `Train Macro`：辅助任务训练损失
- `Train Total`：加权组合损失
- `Valid Main`：主任务验证损失（用于early stopping）
- `Test Main`：当前轮的测试损失
- `Macro Val`：辅助任务验证损失
- `Best Epoch`：目前最佳模型的轮数

#### 7.7 输出与保存

```
1. 检查是否存在最佳预测
   （如果early stop在第1轮，需从测试集重新评估）

2. 添加预测列到测试样本
   raw_test_samples['prediction'] = best_preds

3. 保存预测到CSV
   output_file = {model_dir}/predictions.csv
   格式: TSV, index=False

4. 输出最终结果
   Best epoch: 042
   Best valid loss: 0.2456
   Best test loss: 0.2567
```

---

### 8. `main(config: BaseConfig) → None`

**程序主入口**，支持单fold或多fold模式。

**逻辑**：

```
if config.dataset.synergy 有 test_fold 属性:
    # 单fold模式
    run_fold(config, logger)
else:
    # 多fold交叉验证模式
    for i in range(num_folds):
        config_tmp = deepcopy(config)
        config_tmp.model_dir = {base_dir}/{i}
        config_tmp.dataset.synergy.test_fold = i
        run_fold(config_tmp, logger)
```

**多fold示例**：

```
num_folds = 5
├─ output/0/  ← fold-0作为测试集
│  ├─ configs.yml
│  ├─ train.log
│  └─ predictions.csv
├─ output/1/  ← fold-1作为测试集
├─ output/2/
├─ output/3/
└─ output/4/
```

---

## 主程序入口

```python
if __name__ == "__main__":
    1. 解析命令行参数
       - config: 配置文件路径 (必需)
       - --sd: 随机种子 (default=18)
       - --update: 配置覆盖 (optional, 格式: tasks.synergy.weight=2.0)

    2. 加载配置并应用覆盖
       config = get_default_config(args.config)
       if args.update:
           对每个覆盖: config.set_config_via_path(key, value)

    3. 设置随机种子 (保证可重复性)
       seet_random_seed(args.sd)

    4. 启动训练
       main(config)
```

---

## 使用示例

### 示例 1: 单fold训练

```bash
python train_multitask_synergy_macro.py configs/multitask_synergy_macro.yml \
    --sd 42
```

### 示例 2: 多fold交叉验证

```bash
python train_multitask_synergy_macro.py configs/multitask_synergy_macro.yml \
    --sd 42
```

### 示例 3: 命令行覆盖配置

```bash
python train_multitask_synergy_macro.py configs/multitask_synergy_macro.yml \
    --update tasks.synergy.weight=1.0 \
             tasks.macro_link.weight=0.5 \
             trainer.num_epochs=100 \
    --sd 42
```

---

## 配置文件示例

```yaml
# configs/multitask_synergy_macro.yml
dataset:
  synergy:
    train_folds: [0, 1, 2, 3]
    test_fold: 4
    num_folds: 5
  macro:
    root: data/proc/macro/
    name: macro_heterograph

model:
  synergy:
    seq_max_len: 512
    vocab_sizes: [...]
  macro:
    hidden_size: 256
    num_heads: 8
    num_layers: 2

trainer:
  optimizer:
    lr: 1e-4
    weight_decay: 1e-5
  scheduler:
    name: constant
    params:
      num_warmup_steps: 500
      num_training_steps: 5000
  num_epochs: 100
  patience: 8

tasks:
  synergy:
    enabled: true
    weight: 1.0
  macro_link:
    enabled: true
    weight: 0.2
    every_n_steps: 1

model_dir: output/multitask_r1/
gpu: 0
```

---

## 关键设计决策

### 1. 为何使用多任务学习？

| 优势         | 说明                                         |
| ------------ | -------------------------------------------- |
| **特征共享** | 异构图特征可增强Synergy模型的药物/蛋白质表示 |
| **正则化**   | 辅助任务作为隐式正则化，防止过拟合           |
| **泛化性**   | 异构图约束帮助模型学习更鲁棒的表示           |
| **数据利用** | 充分利用异构图的结构信息                     |

### 2. 为何权重比是 1.0 : 0.2？

- 主任务（Synergy回归）是核心目标 → 权重=1.0
- 辅助任务（图预测）是支援角色 → 权重=0.2（相对较小）
- 防止辅助任务过度主导训练

### 3. 为何每步都计算辅助损失（every_n_steps=1）？

- 对比alternative: every_n_steps=5（每5步计算一次）
- 每步计算 → 更频繁的图约束信号，更好的梯度稳定性
- 计算成本可控（MacroEncoder较小）

### 4. 模型选择标准

- 基于**验证集的主任务损失**（Synergy MSE）
- 而非总损失或图损失
- 原因：主任务是最终评价指标

---

## 依赖关系

```
train_multitask_synergy_macro.py
├─ models.datasets
│  ├─ SynergyDataset: 加载协同性数据
│  └─ MacroNetDataset: 加载异构图
├─ models.models
│  ├─ SynergyBert: 主任务模型
│  └─ MacroEncoder: 辅助任务模型
├─ models.utils
│  ├─ seet_random_seed: 设置随机种子
│  ├─ get_logger: 日志记录
│  ├─ convert_to_bert_config: 配置转换
│  ├─ get_scheduler_by_name: 学习率调度
│  └─ count_model_params: 参数统计
└─ my_config
   └─ BaseConfig: 配置管理
```

---

## 总结

**train_multitask_synergy_macro.py** 实现了一个**多任务学习框架**，通过联合优化：

- ✅ **主任务**：药物协同性预测（Synergy回归）
- ✅ **辅助任务**：异构图链接预测（Macro链接预测）

**主要贡献**：

1. 充分利用异构图的拓扑和语义信息
2. 通过多任务学习提升主任务的泛化性能
3. 支持灵活的配置、多fold交叉验证、早停机制
4. 完整的日志和结果保存功能

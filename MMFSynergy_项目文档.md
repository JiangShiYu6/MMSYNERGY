# MMFSynergy 项目技术文档

> **论文标题**: Fusing Micro- and Macro-Scale Information to Predict Anticancer Synergistic Drug Combinations
>
> **许可证**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

---

## 目录

1. [项目概述](#1-项目概述)
2. [项目结构](#2-项目结构)
3. [环境依赖](#3-环境依赖)
4. [核心思路与整体架构](#4-核心思路与整体架构)
5. [配置系统](#5-配置系统)
6. [数据处理模块](#6-数据处理模块)
7. [模型架构详解](#7-模型架构详解)
8. [训练流程详解](#8-训练流程详解)
9. [推理与特征提取](#9-推理与特征提取)
10. [评估与分析](#10-评估与分析)
11. [运行指南](#11-运行指南)

---

## 1. 项目概述

MMFSynergy 是一个用于**抗癌药物协同效应预测**的深度学习框架。当两种药物联合使用治疗癌症时，它们可能产生协同增效作用（1+1 > 2）。本项目通过融合**微观**和**宏观**两个尺度的生物信息来预测药物组合的协同评分（Synergy Score）。

- **微观信息（Micro-Scale）**：基于药物的 SMILES 分子式和蛋白质的氨基酸（AA）序列，利用 BERT 类模型提取生化特征。
- **宏观信息（Macro-Scale）**：基于药物-蛋白质-副作用之间的异构生物网络，利用图注意力网络（HAN）提取全局拓扑特征。

最终，微观和宏观特征被融合后送入 SynergyBert 模型，进行协同评分的回归预测。

---

## 2. 项目结构

```
MMFSynergy/
├── configs/                          # YAML 配置文件
│   ├── protein_aa_tokenizer.yml      # AA 分词器配置
│   ├── drug_smiles_tokenizer.yml     # SMILES 分词器配置
│   ├── protein_aa_encoder.yml        # AA 编码器（MLM）配置
│   ├── drug_smiles_encoder.yml       # SMILES 编码器（MLM）配置
│   ├── protein_aa_encoder_simcse.yml # AA 编码器（SimCSE）配置
│   ├── drug_smiles_encoder_simcse.yml# SMILES 编码器（SimCSE）配置
│   ├── macro.yml                     # 宏观编码器配置
│   ├── drug_micro_infer.yml          # 药物微观推理配置
│   ├── protein_micro_infer.yml       # 蛋白质微观推理配置
│   ├── macro_infer.yml               # 宏观推理配置
│   ├── fuse_drug.yml                 # 药物特征融合配置
│   ├── fuse_protein.yml              # 蛋白质特征融合配置
│   ├── nested_cv.yml                 # 嵌套交叉验证配置
│   └── multitask_synergy_macro.yml   # 多任务训练配置
│
├── models/                           # 模型定义
│   ├── __init__.py
│   ├── models.py                     # 所有模型类
│   ├── datasets.py                   # 所有数据集类
│   └── utils.py                      # 工具函数
│
├── data/                             # 数据目录
│   ├── proc/                         # 预处理后的数据
│   └── proc_data.py                  # 数据预处理脚本
│
├── output/                           # 模型输出目录
│
├── my_config.py                      # 配置系统基类
├── train_tokenizer.py                # 分词器训练
├── train_encoder_mlm.py              # MLM 预训练
├── train_encoder_simcse.py           # SimCSE 预训练
├── train_main_macro.py               # 宏观编码器训练
├── train_fusion.py                   # 特征融合训练
├── train_multitask_synergy_macro.py  # 多任务联合训练
├── train_micro.py                    # 微观模型训练
├── infer_micro.py                    # 微观特征推理
├── infer_macro.py                    # 宏观特征推理
├── nested_cv.py                      # 嵌套交叉验证
├── assess_macro_data_quality.py      # 宏观数据质量评估
├── plot_quality_curves.py            # 可视化鲁棒性曲线
│
├── run_exp_aa_encoders_mlm.sh        # AA 编码器 MLM 实验脚本
├── run_exp_aa_encoders_simcse.sh     # AA 编码器 SimCSE 实验脚本
├── run_exp_smiles_encoders_mlm.sh    # SMILES 编码器 MLM 实验脚本
├── run_exp_smiles_encoders_simcse.sh # SMILES 编码器 SimCSE 实验脚本
├── run_exp_macro.sh                  # 宏观编码器实验脚本
├── run_exp_fusion.sh                 # 融合实验脚本
├── run_ncv.sh                        # 嵌套交叉验证脚本
│
├── requirements.txt                  # Python 依赖
└── readme.md                         # 项目说明
```

---

## 3. 环境依赖

| 依赖库 | 版本 | 用途 |
|--------|------|------|
| torch | 2.4.0 | 深度学习框架 |
| dgl | 2.4.0+cu124 | 异构图神经网络 |
| transformers | 4.57.0 | BERT 模型实现 |
| tokenizers | 0.13.3 | 快速子词分词 |
| rdkit | 2023.3.2 | 化学信息学（SMILES 处理） |
| PyYAML | 6.0 | 配置文件解析 |
| scikit-learn | 3.6.0 | 机器学习工具 |
| pandas | 2.3.3 | 数据处理 |
| numpy | 2.0.2 | 数值计算 |
| scipy | 1.7.1 | 科学计算 |
| networkx | 3.2.1 | 网络/图分析 |
| tqdm | 4.62.3 | 进度条显示 |

安装命令：
```bash
pip3 install -r requirements.txt
```

---

## 4. 核心思路与整体架构

### 4.1 方法论

本项目的核心理念是：**药物协同效应既取决于分子本身的化学性质（微观），也取决于药物在生物网络中的相互作用关系（宏观）**。

### 4.2 多阶段训练流水线

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Stage 1: 分词器训练                             │
│   AA序列 / SMILES字符串 → BPE/Unigram/WordPiece 分词器              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
┌────────────────────┐                    ┌────────────────────┐
│ Stage 2a: MLM 预训练│                    │ Stage 2c: 宏观编码器│
│ 掩码语言模型训练     │                    │ 异构图注意力网络     │
│ (AA + SMILES)       │                    │ + 链接预测          │
└────────┬───────────┘                    └────────┬───────────┘
         ▼                                         │
┌────────────────────┐                             │
│Stage 2b: SimCSE训练 │                             │
│ 对比学习增强表示     │                             │
└────────┬───────────┘                             │
         │                                         │
         ▼                                         ▼
┌────────────────────┐                    ┌────────────────────┐
│ Stage 3a: 微观推理   │                    │ Stage 3b: 宏观推理  │
│ 提取药物/蛋白质嵌入  │                    │ 提取节点嵌入         │
└────────┬───────────┘                    └────────┬───────────┘
         │                                         │
         └─────────────────┬───────────────────────┘
                           ▼
                ┌────────────────────┐
                │  Stage 4: 特征融合  │
                │  微观 + 宏观 → 融合 │
                └────────┬───────────┘
                         ▼
                ┌────────────────────┐
                │ Stage 5: 协同预测   │
                │ SynergyBert 回归   │
                │ (可选多任务联合训练) │
                └────────┬───────────┘
                         ▼
                ┌────────────────────┐
                │  Stage 6: 评估     │
                │ 嵌套交叉验证       │
                │ 数据质量分析        │
                └────────────────────┘
```

---

## 5. 配置系统

### 5.1 BaseConfig 类（`my_config.py`）

项目使用自定义的 `BaseConfig` 类管理配置，该类继承自 Python `dict`，提供以下特性：

| 特性 | 说明 | 示例 |
|------|------|------|
| YAML 加载/保存 | 从 YAML 文件读写配置 | `config.load_from_file('config.yml')` |
| 点表示法访问 | 像访问属性一样访问配置 | `config.model.hidden_size` |
| 路径式访问 | 通过字符串路径访问嵌套配置 | `config.get_attr_via_path('model.hidden_size')` |
| 路径式设置 | 通过字符串路径设置值 | `config.set_config_via_path('model.lr', 1e-4)` |
| 嵌套字典合并 | 自动递归合并嵌套配置 | `config.update(new_config)` |
| 自动类型转换 | 嵌套字典自动转为 BaseConfig | `config.d = {'a': 1}` → BaseConfig 对象 |

### 5.2 配置文件结构

典型的配置文件包含以下部分：

```yaml
dataset:                    # 数据集配置
  samples: data/path.tsv    # 数据文件路径
  train/valid/test:
    loader:
      batch_size: 256
      shuffle: true
      pin_memory: true

model:                      # 模型配置
  hidden_size: 384
  num_attention_heads: 8
  num_hidden_layers: 3

trainer:                    # 训练器配置
  num_epochs: 500
  patience: 100
  optimizer:
    lr: 2.e-4
    weight_decay: 1.e-3
  scheduler:
    name: constant_with_warmup
    params:
      num_warmup_steps: 2000

gpu: 0                      # GPU 设备编号
model_dir: output/path      # 输出目录
```

---

## 6. 数据处理模块

### 6.1 数据集类概览（`models/datasets.py`）

| 数据集类 | 用途 | 输入数据 | 输出格式 |
|----------|------|----------|----------|
| `TextDatasetForMLM` | MLM 预训练 | 文本文件（AA/SMILES序列） | input_ids, attention_mask, mlm_labels |
| `TextDatasetForSimCSE` | SimCSE 对比学习 | 文本文件 | 重复的 input_ids 对 + 对比标签 |
| `MacroNetDataset` | 宏观图网络 | TSV 边列表 + 节点特征 | DGL 异构图 + 边划分 |
| `SynergyDataset` | 协同预测主任务 | TSV 样本表 | drug_ids, protein_ids, weights, labels |
| `MicroInferDataset` | 微观推理 | 索引 + 文本序列 | sample_indices, input_ids, attention_mask |
| `FusionDataset` | 特征融合 | 微观/宏观嵌入张量 | 正负样本对 + 标签 |

### 6.2 TextDatasetForMLM 详解

用于掩码语言模型预训练，关键参数：

- **mask_rate**: 0.15（15% 的 token 被选中进行掩码）
- **掩码策略**：被选中的 token 中：
  - 80% 替换为 `[MASK]` token
  - 10% 替换为随机 token
  - 10% 保持不变
- **collate_fn**: 将不等长序列填充到批次内最长长度，生成 attention_mask

### 6.3 TextDatasetForSimCSE 详解

用于对比学习，关键机制：

- 每个样本生成**正样本对**：同一序列复制两次 `(x, x)`
- 利用 BERT 的 dropout 机制产生不同的表示
- 批内其他样本作为**负样本**
- 标签：`torch.arange(batch_size)`，用于对比损失计算

### 6.4 MacroNetDataset 详解

基于 DGL 的异构图数据集：

- **节点类型**：drug（药物）、protein（蛋白质）、sideeffect（副作用）
- **边类型**：
  - `drug2drug`（DDI，药物-药物相互作用）
  - `drug2protein` / `protein2drug`（DTI，药物-靶标相互作用）
  - `protein2protein`（PPI，蛋白质-蛋白质相互作用）
  - `sideeffect2drug`（DSI，副作用-药物关联）
- **加载文件**：`drug2idx.tsv`, `protein2idx.tsv`, `sideeffect2idx.tsv`（索引映射），`ddi.tsv`, `dti.tsv`, `ppi.tsv`, `dsi.tsv`（边列表），`*_feat.npy`（节点特征）
- **边划分**：每种边类型按 5%/5% 分为 val/test 集

### 6.5 SynergyDataset 详解

协同预测主任务的数据集：

- **输入格式**：TSV 文件，列包含 `drug_row_idx`, `drug_col_idx`, `cell_line_idx`, `fold`, `synergy_score`
- **细胞系-蛋白质映射**：每个细胞系关联一组蛋白质及其权重，按绝对值排序
- **pad_batch 方法**：
  - 最大序列长度 256（2 个药物 + 最多 254 个蛋白质）
  - 返回包含 `drug_comb_ids`, `protein_ids`, `weights`, `attention_mask`, `labels` 的字典
  - 蛋白质和权重填充到批次内最大数量

---

## 7. 模型架构详解

### 7.1 微观编码器

#### 7.1.1 BertWithoutSegEmb

去掉 segment embedding 的 BERT 模型。标准 BERT 使用 segment embedding 区分句子对，但本项目处理的是单一序列（AA 或 SMILES），因此不需要。

```
输入序列 → Token Embedding + Position Embedding → LayerNorm → Dropout → BertEncoder → BertPooler
```

**变体**：

| 类名 | 预训练任务 | 损失函数 |
|------|-----------|----------|
| `BertWithoutSegEmbForMaskedLM` | 掩码语言模型 | CrossEntropyLoss（忽略标签为 -100 的位置） |
| `BertWithoutSegEmbForSimCSE` | 对比学习 | CrossEntropyLoss（基于余弦相似度矩阵） |

#### 7.1.2 SimCSE 组件

- **SimCSEPooler**：支持 5 种池化策略
  - `cls`：取 [CLS] token 经 MLP 后的表示
  - `cls_before_pooler`：直接取 [CLS] token
  - `avg`：所有 token 的平均
  - `avg_top2`：最后两层隐层的平均
  - `avg_first_last`：首尾两层隐层的平均
- **Similarity**：带温度缩放的余弦相似度 `cos_sim(z1, z2) / temperature`

### 7.2 宏观编码器（MacroEncoder）

基于异构注意力网络（HAN）的图神经网络：

```
节点特征 → 线性投影层（per node type） → HANLayer ×N → 节点嵌入
                                                         ↓
                                              链接预测头（per edge type）
                                                         ↓
                                                    BCELoss
```

#### 7.2.1 HANLayer

每个 HAN 层包含两级注意力：

1. **节点级注意力**：使用 GATConv（图注意力卷积）对每种边类型分别计算
2. **语义级注意力**：对不同边类型产生的结果进行注意力加权聚合

```python
# 节点级注意力（per edge type）
h_etype = GATConv(h_src, h_dst)  # 每种边类型一个 GAT

# 语义级注意力
attention_weights = softmax(tanh(Linear(mean(h_etypes))))
h_final = sum(attention_weights * h_etypes)
```

#### 7.2.2 链接预测头

每种边类型有一个链接预测头：
```
concat(src_emb, dst_emb) → Linear → ReLU → Linear → Sigmoid → BCELoss
```

**负采样**：随机采样等量负样本边（1:1 比例），动态生成以保证多样性。

**默认超参数**：

| 参数 | 值 |
|------|-----|
| HAN 层数 | 2 |
| 隐藏维度 | 128 |
| 注意力头数 | 8 |
| Dropout | 0.2 |

### 7.3 协同预测模型（SynergyBert）

主模型，用于预测药物组合的协同评分：

```
Drug A idx ─┐
Drug B idx ─┤
             ├→ DrugProteinEmbeddingLayer → BertEncoder (3层) → BertSynergyPooler → BertHeadForSynergy → Synergy Score
Protein ids ─┤
Weights ─────┘
```

#### 7.3.1 DrugProteinEmbeddingLayer

将药物和蛋白质嵌入到统一空间：

- **药物嵌入**：从预计算的微观特征初始化（`frozen=True`，训练时不更新）
- **蛋白质嵌入**：同样从预训练特征初始化
- **权重嵌入**：细胞系特异性的蛋白质权重
- **序列构造**：`[Drug_A, Drug_B, Protein_1, Protein_2, ..., Protein_P]`
- 所有嵌入通过线性层投影到统一的 `hidden_size`，最后经过 LayerNorm + Dropout

#### 7.3.2 BertSynergyPooler

自定义池化层：取序列前 2 个 token（即两个药物的表示）的平均值，经过 Dense + Tanh 激活。

#### 7.3.3 BertHeadForSynergy

回归头：`Linear(hidden_size → 1)`，输出单个协同评分。

**默认超参数**：

| 参数 | 值 |
|------|-----|
| hidden_size | 384 |
| drug_size | 38 |
| drug_hidden_size | 256 |
| protein_size | 4871 |
| protein_hidden_size | 256 |
| num_attention_heads | 8 |
| num_hidden_layers | 3 |
| intermediate_size（FFN 隐层） | 1536 |
| dropout | 0.1 |
| freeze_embedding | True |

### 7.4 融合模型（FusionModel）

将微观和宏观特征融合为统一表示：

```
micro_feat → Linear → ┐
                       ├→ concat → Linear → ReLU → Dropout → Linear → Sigmoid
macro_feat → Linear → ┘
```

### 7.5 自编码器（AutoEncoder）

用于特征压缩/降维：

```
输入 → Encoder(Linear → ReLU → Dropout → Linear) → Latent → Decoder(Linear → ReLU → Dropout → Linear) → 重建
```

损失函数：L1Loss（绝对值损失）

---

## 8. 训练流程详解

### 8.1 Stage 1: 分词器训练（`train_tokenizer.py`）

支持 4 种子词分词算法：

| 算法 | 说明 |
|------|------|
| BPE | 字节对编码，最常用 |
| Unigram | 基于一元语言模型 |
| WordPiece | BERT 原始使用的算法 |
| WordLevel | 词级别分词 |

**特殊 token**：

| Token | 用途 |
|-------|------|
| [UNK] | 未知 token |
| [CLS] | 分类 token（序列首） |
| [SEP] | 分隔 token（序列尾） |
| [PAD] | 填充 token |
| [MASK] | 掩码 token（MLM 使用） |

**输出**：
- `tokenizer.json`：分词器权重
- `config.json`：特殊 token ID 和词表大小

### 8.2 Stage 2a: MLM 预训练（`train_encoder_mlm.py`）

训练 `BertWithoutSegEmbForMaskedLM` 模型：

```
输入序列 → 15% token 被掩码 → BERT 编码器 → 预测被掩码位置的原始 token
```

**训练流程**：
1. 加载分词器和数据集
2. 构建 BERT 模型
3. 使用 AdamW 优化器 + 学习率调度器
4. 逐批次训练，定期在验证集上评估
5. 早停机制：验证损失不再下降时停止
6. 保留 Top-K 检查点

**关键配置**：
- 最大序列长度：512
- 调度器：constant / linear / cosine / polynomial / inverse_sqrt
- 早停耐心值：由配置文件指定

### 8.3 Stage 2b: SimCSE 预训练（`train_encoder_simcse.py`）

训练 `BertWithoutSegEmbForSimCSE` 模型：

```
同一序列输入两次 → Dropout 产生不同表示 (z1, z2) → 余弦相似度矩阵 → 对比损失
```

**特点**：
- 可加载 MLM 预训练权重作为初始化（通过 `pretrain_model_path` 配置）
- 仅使用训练集（验证/测试已注释掉）
- 利用 BERT 内置的 Dropout 作为数据增强

### 8.4 Stage 2c: 宏观编码器训练（`train_main_macro.py`）

训练 `MacroEncoder` 在异构图上进行链接预测：

```
异构图 → HANLayer ×2 → 节点嵌入 → 链接预测（每种边类型）→ BCELoss
```

**训练细节**：
- 正样本：图中存在的边
- 负样本：随机采样的不存在的边（1:1 比例）
- 分别对 4 种边类型计算 BCE 损失后取平均
- 定期在验证集边上评估

### 8.5 Stage 5: 多任务联合训练（`train_multitask_synergy_macro.py`）

同时训练 SynergyBert 和 MacroEncoder：

```
总损失 = 1.0 × MSE(协同预测) + 0.2 × BCE(宏观链接预测)
```

**任务配置**：

| 参数 | 值 | 说明 |
|------|-----|------|
| synergy.weight | 1.0 | 主任务权重 |
| macro_link.weight | 0.2 | 辅助任务权重 |
| macro_link.every_n_steps | 1 | 辅助任务执行频率 |

**训练流程**：
1. 创建协同数据加载器（训练/验证/测试）
2. 加载宏观异构图
3. 每步先计算协同预测的 MSE 损失
4. 按频率计算宏观链接预测的 BCE 损失
5. 加权求和后反向传播
6. 基于验证集的协同 MSE 损失做模型选择
7. 早停耐心值：30

**与单任务训练的区别**：
- 使用验证集（而非测试集）做早停
- 两个模型联合优化
- 宏观任务提供正则化效果

### 8.6 Stage 4: 特征融合训练（`train_fusion.py`）

训练 `FusionModel` 融合微观和宏观表示，结构与主训练脚本类似。

---

## 9. 推理与特征提取

### 9.1 微观推理（`infer_micro.py`）

从预训练的 SimCSE 编码器提取药物/蛋白质嵌入：

```
AA/SMILES 序列 → 加载预训练 BertWithoutSegEmb → 取 [CLS] token 的隐层表示 → 保存为嵌入矩阵
```

**流程**：
1. 加载预训练模型配置和权重
2. 遍历所有药物/蛋白质序列
3. 提取 `last_hidden_state[:, 0, :]`（CLS 向量）
4. 构建嵌入矩阵（n_entities × hidden_dim）
5. 保存为 `.pth` 文件

### 9.2 宏观推理（`infer_macro.py`）

从预训练的 MacroEncoder 提取节点嵌入：

```
异构图 → 加载预训练 MacroEncoder → 前向传播 → 保存 drug/protein 嵌入
```

**输出**：分别为 drug 和 protein 保存嵌入向量。

---

## 10. 评估与分析

### 10.1 嵌套交叉验证（`nested_cv.py`）

采用嵌套 CV 实现无偏的超参数选择和性能评估：

```
外层循环（5 折）: 用于最终性能评估
  ├── 内层循环（每折内 4 个子折）:
  │   ├── 遍历所有超参数组合
  │   ├── 每种组合在每个验证子折上训练 + 评估
  │   └── 选择验证损失最小的超参数
  └── 用最佳超参数在所有内层数据上训练
      └── 在外层测试折上评估
```

**超参数搜索空间**（默认配置）：

| 超参数 | 候选值 |
|--------|--------|
| num_warmup_steps | [3000, 2000] |
| learning_rate | [2e-4, 1e-4] |

总计 2 × 2 = 4 种组合。

**并行计算**：支持多 GPU 并行（默认 8 GPU），使用 `multiprocessing` + `Queue` 实现 GPU 资源调度。

**任务规模**：
- 内层任务数 = 5折 × 4子折 × 4超参组合 = 80 次训练
- 外层任务数 = 5 次训练
- 总计 85 次训练

### 10.2 数据质量评估（`assess_macro_data_quality.py`）

通过扰动分析评估宏观网络数据的鲁棒性：

- **边扰动**：按比例随机删除边，观察链接预测性能变化
- **节点扰动**：按比例随机删除节点，观察性能变化
- **评估指标**：AUC、AP（Average Precision）
- **输出**：CSV 格式的质量评估报告

### 10.3 可视化（`plot_quality_curves.py`）

绘制鲁棒性曲线，展示在不同扰动率下 AUC 和 AP 的变化趋势。

---

## 11. 运行指南

### 11.1 完整训练流程

```bash
# Step 1: 训练分词器
python3 train_tokenizer.py configs/protein_aa_tokenizer.yml --train
python3 train_tokenizer.py configs/drug_smiles_tokenizer.yml --train

# Step 2a: 训练微观编码器（MLM）
bash run_exp_aa_encoders_mlm.sh
bash run_exp_smiles_encoders_mlm.sh

# Step 2b: 训练微观编码器（SimCSE）
bash run_exp_aa_encoders_simcse.sh
bash run_exp_smiles_encoders_simcse.sh

# Step 2c: 训练宏观编码器
bash run_exp_macro.sh

# Step 3: 提取嵌入特征
python3 infer_micro.py configs/drug_micro_infer.yml
python3 infer_micro.py configs/protein_micro_infer.yml
python3 infer_macro.py configs/macro_infer.yml

# Step 4: 特征融合
bash run_exp_fusion.sh

# Step 5: 嵌套交叉验证评估
bash run_ncv.sh
```

### 11.2 命令行参数覆盖

所有训练脚本支持通过 `-u` 参数动态覆盖配置：

```bash
python3 train_encoder_mlm.py configs/protein_aa_encoder.yml \
    -u trainer.optimizer.lr=1e-4 \
    -u model.hidden_size=256 \
    -u gpu=1
```

### 11.3 输出结构

```
output/
├── pretrain_micro/           # 微观预训练输出
│   ├── model_*.pt            # 模型检查点
│   ├── configs.yml           # 训练配置
│   └── train.log             # 训练日志
├── pretrain_macro/           # 宏观预训练输出
├── v7/                       # 主实验输出
│   ├── ncv/                  # 嵌套交叉验证
│   │   └── predictions-*.csv # 每折预测结果
│   └── multitask_synergy_macro/  # 多任务训练
└── ...
```

---

## 附录：工具函数说明（`models/utils.py`）

| 函数 | 功能 |
|------|------|
| `seet_random_seed(seed=18)` | 设置全局随机种子（torch, numpy, random, 环境变量） |
| `get_logger(name, file)` | 创建带文件/控制台处理器的 Logger |
| `set_log(file, queue)` | 多进程日志：设置 QueueListener |
| `queue_log(queue)` | 多进程日志：添加 QueueHandler |
| `close_log()` | 停止日志监听器 |
| `convert_to_bert_config(config)` | 将自定义配置转为 HuggingFace BertConfig |
| `get_scheduler_by_name(name, optimizer)` | 学习率调度器工厂函数 |
| `keep_top_k_checkpoints(ckpts, k, cmp)` | 保留 Top-K 检查点，删除过期检查点 |
| `kv_args(arg)` | 解析 `key=value` 格式的命令行参数 |
| `count_model_params(model)` | 统计模型参数量（总数/可训练/冻结） |
| `random_split_indices(dataset, test_size)` | 按唯一键分层划分数据集 |

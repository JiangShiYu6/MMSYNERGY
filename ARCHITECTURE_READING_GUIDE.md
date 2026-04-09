# MMFSynergy 架构阅读文档（面向多任务学习改造）

## 1. 文档目标

这份文档帮助你快速理解当前仓库的整体结构、数据流与训练流程，并明确后续改造成多任务学习框架时的切入点。

你可以把当前代码理解为一个“分阶段特征工程 + 终端协同预测”的流水线：

1. 微观阶段：从 SMILES/AA 文本预训练编码器，得到药物/蛋白微观向量。
2. 宏观阶段：在异构图上训练宏观编码器，得到药物/蛋白宏观向量。
3. 融合阶段：融合微观和宏观向量，作为下游协同预测特征。
4. 终端任务：SynergyBert 根据 drugA/drugB + 细胞相关蛋白特征做协同评分回归。

## 2. 项目结构速览

- 顶层训练/推理脚本
  - train_tokenizer.py：训练 BPE/Unigram/WordPiece 分词器。
  - train_encoder_mlm.py：微观编码器 MLM 预训练。
  - train_encoder_simcse.py：微观编码器 SimCSE 训练。
  - infer_micro.py：微观编码器推理，导出实体 embedding。
  - infer_macro.py：宏观编码器推理，导出实体 embedding。
  - train_main_macro.py：当前名称是 macro，但内容实际上是 SynergyBert 训练入口。
  - train_fusion.py：当前实现与 train_main_macro.py 基本一致，也是在训练 SynergyBert。
  - nested_cv.py：SynergyBert 的嵌套交叉验证。
- 核心模块
  - models/models.py：模型定义（微观编码器、宏观编码器、融合模块、终端预测模型）。
  - models/datasets.py：文本数据集、协同样本数据集、异构图数据集等。
  - models/utils.py：日志、随机种子、scheduler、配置转换、checkpoint 管理。
  - my_config.py：层级化配置容器 BaseConfig。
- 配置目录
  - configs/\*.yml：覆盖分词器、预训练、推理、融合、NCV。
- 数据预处理
  - data/proc_data.py：数据加工实验脚本集合（历史版本较多，偏离主流程但可查来源）。

## 3. 端到端数据流（概念图）

```text
Raw text/network data
  -> tokenizer training (AA/SMILES)
  -> micro encoder pretrain (MLM/SimCSE)
  -> infer_micro => drug_micro_feat.pth / protein_micro_feat.pth

Raw heterograph data
  -> macro encoder training (HAN/GAT on heterograph)
  -> infer_macro => drug_macro_feat.pth / protein_macro_feat.pth

Micro + Macro features
  -> fusion stage (intended: FusionModel)
  -> drug/protein fused features

Drug pair + cell-associated proteins + feature files
  -> SynergyBert
  -> synergy score regression
  -> nested CV / per-fold predictions
```

## 4. 关键模型职责

## 4.1 终端协同预测：SynergyBert

位置：models/models.py

核心思想：

1. DrugProteinEmbeddingLayer
   - 输入：drug_comb_ids (batch, 2)、protein_ids (batch, P)、weights (batch, P)。
   - 从外部 npy 文件加载药物/蛋白初始向量（可冻结）。
   - 通过 projector 对齐到统一 hidden_size。
   - 将药物双塔和蛋白序列拼接为 token 序列。
2. BertEncoder
   - 对拼接序列做多层自注意力编码。
3. BertSynergyPooler + BertHeadForSynergy
   - 池化 drugA/drugB 区域后回归输出协同分值。

这部分是当前仓库最可运行、最完整的主任务链路。

## 4.2 微观编码器：BertWithoutSegEmb 系列

位置：models/models.py

- BertWithoutSegEmb：去掉 segment embedding 的 BERT 主体。
- BertWithoutSegEmbForMaskedLM：MLM 训练头。
- BertWithoutSegEmbForSimCSE：对比学习头（温度缩放、pooler 可配）。

用途：

- 对 SMILES/AA 的语义建模，最终在 infer_micro.py 导出实体向量。

## 4.3 宏观编码器：MacroEncoder

位置：models/models.py

- 使用 HeteroGraphConv + GATConv + semantic attention。
- 输入是异构图（drug/protein/sideeffect 节点，多关系边）。
- 输出每种节点类型的表示。
- 提供 link_pred_loss，支持边预测式训练目标。

用途：

- 学习网络拓扑层面的宏观表示，infer_macro.py 导出 drug/protein 向量。

## 4.4 融合模型：FusionModel

位置：models/models.py

- 将 micro 与 macro 分别线性投影，拼接后分类器输出匹配概率。
- 与 FusionDataset 对应，数据构造包含正负配对。

注意：当前顶层 train_fusion.py 并没有调用 FusionModel，属于“代码能力已定义，但入口脚本未对齐”的状态。

## 5. 数据集与样本组织

位置：models/datasets.py

- SynergyDataset
  - 从 samples.tsv 读取 drug_row_idx、drug_col_idx、cell_line_idx、fold、synergy_xxx。
  - 从 cell_protein_association 读取每个细胞对应蛋白及权重。
  - pad_batch 会组装为 SynergyBert 需要的 batch 张量。
- TextDatasetForMLM / TextDatasetForSimCSE
  - 负责文本编码、动态 masking、对比学习 batch 拼接。
- MacroNetDataset
  - 从 drug/protein/sideeffect 与多关系网络文件建异构图。
  - 生成 train/val/test 边划分。
- MicroInferDataset
  - 对实体文本做编码并按 idx 回填 embedding。
- FusionDataset
  - 同一样本的 micro 与 macro 做正例，micro 与随机 macro 做负例。

## 6. 配置系统与可复用机制

位置：my_config.py

BaseConfig 支持：

1. 点路径更新：path.to.key=value。
2. 递归 merge：默认配置 + yml 覆盖 + CLI 覆盖。
3. 字典/属性双访问。

这让脚本可以通过 -u 参数快速做超参网格实验，是后续多任务统一配置的基础。

## 7. 当前“主路径”与“代码不一致点”

从脚本和配置对照看，建议你按下面理解：

1. 主路径（相对一致）
   - train_tokenizer.py -> train_encoder_mlm.py / train_encoder_simcse.py -> infer_micro.py
   - nested_cv.py（SynergyBert 任务）
2. 可疑/未对齐路径
   - run_exp_macro.sh 调用 train_macro.py，但仓库里没有 train_macro.py。
   - train_main_macro.py 和 train_fusion.py 实际内容几乎相同，且都训练 SynergyBert。
   - configs/fuse\_\*.yml 是给 FusionModel 用的，但 train_fusion.py 当前未按该意图实现。
   - readme.md 提到 run_exp_fusion，但未明确当前实现偏差。

结论：仓库处于“研究代码演进中间态”，模型定义层比脚本编排层更完整。

## 8. 建议阅读顺序（高效）

1. readme.md：先建立实验阶段概念。
2. models/models.py：看四类模型（SynergyBert、微观编码器、MacroEncoder、FusionModel）。
3. models/datasets.py：看每阶段数据如何喂给模型。
4. train_encoder_mlm.py + train_encoder_simcse.py + infer_micro.py：明确微观阶段产物。
5. infer_macro.py + MacroNetDataset：明确宏观图表示产物。
6. nested_cv.py：看最终主任务评估流程。
7. configs/\*.yml：将“代码逻辑”映射为“可运行参数”。

## 9. 面向多任务学习改造：可落地方案

你要做多任务学习时，建议不要直接在现有脚本上叠补丁，而是先做最小重构。

## 9.1 统一任务定义

建议把任务拆成 3 类：

1. 主任务：协同分值回归（现有 SynergyBert 目标）。
2. 辅助任务 A：宏观图 link prediction（MacroEncoder 自带）。
3. 辅助任务 B：微观对比学习或 MLM（冻结或半冻结微观 encoder）。

统一形式：

Loss_total = lambda_main _ Loss_synergy + lambda_macro _ Loss_link + lambda_micro \* Loss_contrast

## 9.2 模型层重组建议

新增一个统一模型壳（例如 MultiTaskSynergyModel）：

1. 复用 DrugProteinEmbeddingLayer + SynergyBert backbone 作为主干。
2. 挂接 MacroEncoder 分支（可共享 drug/protein projection 层）。
3. 挂接 micro encoder 分支（可选择 frozen/finetune）。
4. 统一 forward 返回 dict：
   - predictions
   - loss_main
   - loss_macro
   - loss_micro
   - loss_total

## 9.3 数据层重组建议

当前数据加载是按任务分离的。多任务阶段可采用“两级 batch”策略：

1. 主 batch：SynergyDataset。
2. 辅 batch：按步长间隔采样 MacroNetDataset/TextDataset。

实现方法：

- 在 trainer 中维护多个 iterator。
- 每个 step 先跑主任务，再按频率附加辅任务更新（例如每 2 step 做一次 link prediction）。

## 9.4 训练器重构建议

把现有多个脚本收敛到单一训练入口，例如：

- train_multitask.py
  - build_dataloaders()
  - build_model()
  - compute_losses()
  - train_one_epoch()
  - evaluate()

并保留旧脚本做兼容壳，内部转调新入口，避免配置与历史命令失效。

## 9.5 配置建议

建议新增统一配置层次：

```yaml
tasks:
  synergy:
    enabled: true
    weight: 1.0
  macro_link:
    enabled: true
    weight: 0.2
  micro_contrast:
    enabled: false
    weight: 0.1

train:
  strategy: alternating # or joint
  alt_steps:
    macro_link_every: 2
    micro_contrast_every: 4
```

## 10. 你可以直接用的改造检查清单

1. 先修复脚本命名/入口不一致（尤其 train_macro.py 缺失问题）。
2. 确定 FusionModel 是否保留为独立阶段，还是并入主模型端到端。
3. 将 models 与 datasets 的“任务能力”转成统一 trainer 的多头损失。
4. 给每个任务单独记录 metric（主任务 RMSE，宏观任务 AUC，微观任务对比 loss）。
5. 保持 nested CV 外壳不动，仅替换内部训练函数，先保证可复现。

## 11. 最小可运行路线（建议）

如果你想尽快落地多任务版本，建议按这个顺序做：

1. 新建 train_multitask.py，仅先接入主任务 + macro_link 两任务。
2. 跑一个 fold 验证日志和 loss 曲线是否稳定。
3. 再接入 micro 辅助任务，先 frozen encoder，再尝试微调。
4. 最后把 nested_cv.py 的 run_fold 替换成新训练接口。

这样风险最小，且每一步都有可观测收益。

---

如果你愿意，我下一步可以直接给你生成“多任务训练骨架代码”（模型壳 + trainer + 配置模板），并尽量复用你当前的 BaseConfig 和数据类，减少重写成本。

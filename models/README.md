# models 目录阅读笔记

这份笔记专门帮助你快速理解 models 目录中每个文件负责什么。

## 1. 文件总览

- **init**.py
  - 目前是空文件。
  - 作用主要是把 models 目录标记为 Python 包，方便 import。

- models.py
  - 负责“模型结构定义”。
  - 包含主任务模型（SynergyBert）、微观文本编码器（MLM/SimCSE）、宏观图编码器（MacroEncoder）、融合模型（FusionModel）等。

- datasets.py
  - 负责“数据集与 batch 组织”。
  - 将原始 tsv/txt/图数据转换成训练脚本可直接消费的张量字典。

- utils.py
  - 负责“训练辅助工具”。
  - 包括随机种子、日志、scheduler 构造、checkpoint 保留策略、参数解析等通用函数。

## 2. models.py 重点内容

可以把 models.py 看成四块：

1. 协同预测主模型
   - DrugProteinEmbeddingLayer
     - 把 drug/protein 的预计算特征加载为 embedding，并投影到统一维度。
     - 输入：drug 组合索引、protein 索引、可选权重。
   - SynergyBert
     - 用 BertEncoder 对 [drugA, drugB, proteins...] 序列做编码。
     - 输出：协同分值（回归）。
   - BertHeadForSynergy / BertSynergyPooler
     - 分别是预测头和池化层。

2. 微观文本编码器
   - BertWithoutSegEmb
     - 自定义无 segment embedding 的 BERT backbone。
   - BertWithoutSegEmbForMaskedLM
     - MLM 任务头，输出 MaskedLMOutput。
   - BertWithoutSegEmbForSimCSE
     - SimCSE 对比学习头，输出相似度矩阵和对比损失。
   - SimCSEPooler / Similarity / MLPLayer
     - 分别负责池化、温度缩放余弦相似度、CLS MLP 头。

3. 宏观图编码器
   - HANLayer
     - 在异构图上做多关系 GAT + 语义注意力聚合。
   - MacroEncoder
     - 堆叠多个 HANLayer，输出 drug/protein/sideeffect 节点表示。
     - 带 link_pred_loss，可用于边预测式自监督。

4. 融合与其他模块
   - FusionModel
     - 把 micro/macro 特征投影后拼接，输出融合表示和匹配概率。
   - AutoEncoder
     - 一个独立的简单自编码器模块（用于重构目标）。

## 3. datasets.py 重点内容

datasets.py 的核心价值是“把不同任务的数据形式统一成 DataLoader 批次”。

- SynergyDataset
  - 服务 SynergyBert 主任务。
  - 从样本表读取 drug_row_idx、drug_col_idx、cell_line_idx、synergy 标签。
  - 再从 cell-protein 关联表取 protein 列表和权重。
  - pad_batch 输出：drug_comb_ids、protein_ids、weights、attention_mask、labels。

- TextDatasetForMLM
  - 服务 MLM 预训练。
  - 动态 mask token，构造 labels（未 mask 位置为 -100）。

- TextDatasetForSimCSE
  - 服务 SimCSE。
  - 组织成成对输入，构造对比学习 labels。

- MacroNetDataset
  - 服务宏观图模型。
  - 从 ddi/dti/ppi/dsi 等文件构造 dgl 异构图。
  - 提供 train/val/test 边划分，用于 link prediction。

- MicroInferDataset
  - 服务微观推理阶段。
  - 输入实体 idx + 文本，输出编码后的 input_ids/attention_mask，推理后可回填 embedding 到对应 idx。

- FusionDataset
  - 服务融合训练。
  - 读取 micro/macro 特征，构造正负样本对（正：同 idx；负：随机错配）。

## 4. utils.py 重点内容

utils.py 是训练脚本复用最多的工具层：

- seet_random_seed
  - 统一设置 Python/NumPy/PyTorch 随机种子。

- get_logger / set_log / queue_log / close_log
  - 单进程和多进程训练日志支持。

- convert_to_bert_config
  - 把项目自定义配置对象转成 HuggingFace BertConfig。

- get_scheduler_by_name
  - 按名字创建学习率调度器（constant、linear、cosine 等）。

- keep_top_k_checkpoints
  - 按指标只保留 top-k checkpoint，并删除过期文件。

- kv_args
  - 解析命令行 -u k=v 覆盖配置参数。

- count_model_params
  - 统计总参数量、可训练参数量、冻结参数量。

- random_split_indices
  - 按样本 key 做随机拆分，避免同 key 泄漏到不同集合。

## 5. 结合训练脚本怎么理解

- 如果你看 train_encoder_mlm.py / train_encoder_simcse.py：
  - 重点对应 models.py 里的 BertWithoutSegEmbForMaskedLM / BertWithoutSegEmbForSimCSE
  - 重点对应 datasets.py 里的 TextDatasetForMLM / TextDatasetForSimCSE

- 如果你看 infer_micro.py：
  - 重点对应 BertWithoutSegEmb + MicroInferDataset

- 如果你看 infer_macro.py：
  - 重点对应 MacroEncoder + MacroNetDataset

- 如果你看 nested_cv.py、train_main_macro.py（当前实际是主任务训练）：
  - 重点对应 SynergyBert + SynergyDataset

## 6. 给你后续做多任务改造的定位建议

建议先从这三处改起：

1. models.py
   - 增加一个统一多任务模型壳，组合 SynergyBert 主干 + MacroEncoder 分支 + 可选微观分支。

2. datasets.py
   - 设计多任务 batch 采样策略（主任务批次 + 辅任务批次）。

3. utils.py
   - 增加多任务 loss 记录与日志模板（每个任务单独统计）。

这样你会在最少改动下，把当前“分阶段代码”平滑过渡到“多任务训练框架”。

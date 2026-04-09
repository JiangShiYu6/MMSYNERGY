# MMFSynergy 多任务学习改造落地文档

## 1. 先回答你的问题：难不难

不算简单，但也不是“推倒重来”级别。

对当前仓库来说，难点主要在工程整合，不在基础模型能力：

1. 模型能力已经有雏形：有 SynergyBert、MacroEncoder、FusionModel。
2. 数据能力已经有基础：有 SynergyDataset、MacroNetDataset、文本数据集。
3. 真正缺的是“统一多任务训练入口”和“统一损失管理”。

结论：

- 技术风险：中等。
- 工程工作量：中等偏上。
- 最佳策略：分 3 阶段增量改造，不要一次性大改。

## 2. 总体目标与拆解

目标：构建多场景、多源特征融合、多任务学习的药物协同预测框架，提升泛化能力。

建议拆成 3 个可执行目标：

1. 多源特征统一表示：把微观文本特征 + 宏观图特征接到同一训练图里。
2. 多任务联合训练：主任务协同回归 + 辅助任务（图链路预测/模态对齐）。
3. 多场景评估闭环：在未知药物组合/场景切分上稳定验证。

## 3. 文件级改造清单（你要改哪个文件）

下表是最小可用改造路径。

| 优先级 | 文件                            | 需要改成什么                                                                        | 改完后的效果                           |
| ------ | ------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------- |
| P0     | models/models.py                | 新增 MultiTaskSynergyModel（统一 forward 返回 main loss + aux losses + total loss） | 模型层支持真正多任务，而不是分脚本拼接 |
| P0     | models/datasets.py              | 新增/扩展多任务 batch 组织（主任务 batch + 宏观辅助 batch + 可选微观辅助 batch）    | 数据层能同时喂多个任务                 |
| P0     | 新建 train_multitask.py         | 统一训练入口，计算 total loss = Σ(lambda_i \* loss_i)，记录分任务日志               | 训练流程从单任务变成可配置多任务       |
| P0     | 新建 configs/multitask.yml      | 增加任务开关、权重、采样频率、冻结策略                                              | 通过配置切换任务组合，不用改代码       |
| P1     | nested_cv.py                    | 将 run_fold 的模型与训练调用切换为 train_multitask 接口                             | 保留原 NCV 框架同时支持多任务          |
| P1     | models/utils.py                 | 增加多任务日志与指标聚合辅助函数（可选）                                            | 更清晰地监控每个任务收敛               |
| P1     | train_fusion.py                 | 迁移为兼容壳（调用新入口）或重写成真实 FusionModel 训练                             | 避免语义混乱与重复逻辑                 |
| P2     | infer_micro.py / infer_macro.py | 补充产物规范（特征维度、路径校验、版本标记）                                        | 方便多场景复用与回溯                   |

## 4. 每个核心文件“改成什么效果”

## 4.1 models/models.py

改造动作：

1. 保留现有 SynergyBert、MacroEncoder。
2. 新增 MultiTaskSynergyModel：
   - 主头：synergy 回归（MSE 或 L1）。
   - 辅助头 A：macro link prediction（复用 MacroEncoder.link_pred_loss）。
   - 辅助头 B：micro-macro 对齐（对比损失或二分类匹配损失）。
3. forward 返回字典：
   - prediction
   - loss_main
   - loss_macro_link（可选）
   - loss_modal_align（可选）
   - loss_total

完成判定：

- 模型在一次前向中能同时计算多项损失。
- 关闭任一辅助任务不会报错。

## 4.2 models/datasets.py

改造动作：

1. 保留 SynergyDataset 作为主任务数据。
2. 为多任务训练新增联合采样接口（可新建 MultiTaskBatchBuilder 或辅助函数）：
   - 主任务：SynergyDataset batch。
   - 辅助任务 A：MacroNetDataset 的 train 边 batch。
   - 辅助任务 B：微观/宏观特征配对 batch。
3. 让 DataLoader 输出结构稳定，例如：
   - batch['main']
   - batch['macro_link']
   - batch['modal_align']

完成判定：

- 一个训练 step 能拿到多任务所需输入。
- 不开启辅助任务时只返回 main 分支输入。

## 4.3 train_multitask.py（新建）

改造动作：

1. 统一构建 dataloader、model、optimizer、scheduler。
2. 每步训练按配置计算：
   - loss_total = w_main _ loss_main + w_macro _ loss_macro + w_align \* loss_align
3. 日志输出分任务指标与总损失。
4. 验证/测试至少保留主任务指标。

完成判定：

- 训练日志里同时有 loss_main/loss_aux/loss_total。
- 在只开主任务时行为与当前单任务训练一致。

## 4.4 configs/multitask.yml（新建）

建议字段：

1. tasks.synergy.enabled/weight
2. tasks.macro_link.enabled/weight
3. tasks.modal_align.enabled/weight
4. train.strategy（joint 或 alternating）
5. train.aux_frequency（每几步执行一次辅助任务）
6. freeze.micro_encoder / freeze.macro_encoder

完成判定：

- 不改代码即可切换任务组合和权重。

## 4.5 nested_cv.py

改造动作：

1. 保留外层 NCV 并行调度逻辑。
2. 将 run_fold 内部训练过程改为调用 train_multitask 的统一接口。
3. 外层早停与模型选择仍基于主任务指标。

完成判定：

- NCV 全流程可跑。
- 最终输出保持 predictions-fold 文件格式兼容。

## 5. 如何判断哪些点“已经完成”

## A. 代码结构完成

满足以下 4 条可视为结构完成：

1. 有 MultiTaskSynergyModel。
2. 有 train_multitask.py。
3. 有 configs/multitask.yml。
4. nested_cv.py 能切到多任务训练路径。

## B. 训练功能完成

满足以下 4 条可视为功能完成：

1. 日志出现 loss_main、loss_aux、loss_total。
2. 任务开关可独立关闭/开启。
3. 训练可在“主任务 only”与“多任务”间切换。
4. checkpoint 中保存了多任务配置与最佳主任务指标。

## C. 泛化目标完成（你最关心）

建议用以下标准判断：

1. 在未知组合或新场景划分上，主任务指标优于或不差于单任务。
2. 多随机种子下方差下降（更稳定）。
3. 过拟合迹象减弱（训练-验证 gap 缩小）。

## 6. 三阶段实施计划（推荐）

## 阶段 1：最小多任务版（1-2 天）

1. 新建 train_multitask.py 与 configs/multitask.yml。
2. 先做两任务：主任务 + macro_link。
3. 确保单卡可跑通 1 个 fold。

里程碑：能看到多任务损失同时下降。

## 阶段 2：接入模态对齐（1-2 天）

1. 在 models/models.py 增加 modal_align 头与损失。
2. 在 datasets.py 增加对齐任务 batch。

里程碑：loss_align 收敛，主任务不退化。

## 阶段 3：NCV 全量与对比实验（2-4 天）

1. 接入 nested_cv.py。
2. 跑单任务 vs 多任务对比。
3. 输出结论与超参建议。

里程碑：完成可复现对比报告。

## 7. 风险与规避

1. 风险：任务权重不平衡导致主任务退化。
   - 规避：先固定 w_main=1.0，辅助任务从小权重起步（0.05-0.2）。

2. 风险：辅助任务 batch 太频繁拖慢训练。
   - 规避：隔步更新（例如每 2-4 步更新一次辅助任务）。

3. 风险：多任务后显存上涨。
   - 规避：先冻结微观编码器，后续再逐步解冻。

## 8. 你可以立刻执行的第一步

先做 1 件事：新建 train_multitask.py + configs/multitask.yml，只接入“主任务 + macro_link”。

这是收益最高、风险最低的起点。

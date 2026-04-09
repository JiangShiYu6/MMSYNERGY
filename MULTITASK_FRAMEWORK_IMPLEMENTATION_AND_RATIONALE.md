# MMFSynergy 多任务学习框架实施与方法依据

## 1. 文档目标

本文件回答两个问题：

1. 如何在当前仓库中完成多任务学习框架。
2. 为什么选择该方法（基于当前代码结构与工程约束）。

适用范围：

1. 主任务：药物协同分值回归（SynergyBert）。
2. 辅助任务：宏观异构图链路预测（MacroEncoder.link_pred_loss）。

## 2. 当前代码现状

当前仓库已经具备多任务改造的核心组件，但尚未形成统一训练闭环：

1. 已有主任务模型能力：models/models.py 中的 SynergyBert。
2. 已有宏观图任务能力：models/models.py 中的 MacroEncoder 与 link_pred_loss。
3. 已有主任务数据能力：models/datasets.py 中的 SynergyDataset。
4. 已有宏观图数据能力：models/datasets.py 中的 MacroNetDataset。
5. 缺失点：统一多任务训练入口与统一损失管理。

## 3. 目标框架定义

采用双任务联合优化：

1. 主任务损失：L_main（Synergy 回归，MSE）。
2. 辅助任务损失：L_macro（宏观边预测，BCE）。
3. 总损失：L_total = w_main _ L_main + w_macro _ L_macro。

训练策略：

1. 单优化器联合更新两个子模型参数。
2. 验证与早停由主任务指标驱动（避免偏离最终业务目标）。

## 4. 具体实施步骤（按优先级）

## 第一步：建立多任务训练入口

文件：train_multitask_synergy_macro.py

完成标准：

1. 同时构建 SynergyBert 与 MacroEncoder。
2. 在同一个训练 step 内计算 L_main 与 L_macro。
3. 用权重合成 L_total 并反向传播。
4. 训练日志同时输出 Train Main、Train Macro、Train Total。

## 第二步：配置化任务开关与权重

文件：configs/multitask_synergy_macro.yml

完成标准：

1. 可配置 tasks.synergy.enabled、tasks.macro_link.enabled。
2. 可配置 tasks.synergy.weight、tasks.macro_link.weight。
3. 可配置 tasks.macro_link.every_n_steps（降低辅助任务频率）。

## 第三步：主任务导向验证与保存

文件：train_multitask_synergy_macro.py

完成标准：

1. 验证集以主任务损失作为 best model 判断依据。
2. 保存测试集预测到 predictions.csv。
3. 早停由主任务验证集表现触发。

## 第四步：逐步接入外层评估

文件：nested_cv.py（后续）

完成标准：

1. 保留现有 NCV 调度逻辑。
2. 将 run_fold 内部训练替换为多任务接口。
3. 保持输出文件格式兼容（便于横向对比单任务基线）。

## 5. 为什么选择这个方法（代码级理由）

## 理由 A：最小破坏现有工程结构

1. 现有仓库训练与评估逻辑稳定，尤其 nested_cv.py。
2. 新增独立多任务入口比直接重写旧脚本风险更低。
3. 可以保留现有单任务脚本用于回归测试。

## 理由 B：现有模型能力天然适配该拆分

1. SynergyBert 已是最终监督目标模型，适合作为主任务。
2. MacroEncoder 已提供 link_pred_loss，适合作为辅助任务。
3. 两者都已在当前仓库中实现，复用成本低。

## 理由 C：符合泛化提升路径

1. 主任务负责目标拟合（药物协同分值）。
2. 辅助图任务提供结构先验，减少只依赖单一标签数据的过拟合。
3. 通过共享训练过程，提高对未知组合和新场景的鲁棒性。

## 理由 D：实验可控、可解释

1. 权重可控：通过 w_main 与 w_macro 调节贡献。
2. 频率可控：通过 every_n_steps 控制计算开销。
3. 对比清晰：可直接做单任务与多任务消融实验。

## 6. 推荐实验顺序

1. 先跑主任务 only（w_macro=0）确认与基线一致。
2. 再启用辅助任务（例如 w_macro=0.1 或 0.2）。
3. 调整 every_n_steps（1, 2, 4）观察训练稳定性与速度。
4. 在固定随机种子下做初筛，再扩展到多种子统计均值与方差。

## 7. 完成度验收清单

## 结构验收

1. 存在多任务训练脚本。
2. 存在多任务配置文件。
3. 日志含三类损失字段。

## 功能验收

1. 任务开关可独立关闭/开启。
2. 关闭辅助任务时可退化为单任务训练。
3. 训练可正常导出 predictions.csv。

## 结果验收

1. 与单任务相比，验证/测试主指标不退化。
2. 多次运行结果方差下降或稳定性提升。
3. 新场景或未知组合划分下有可解释收益。

## 8. 风险与规避

1. 风险：辅助任务过强导致主任务下降。
   - 规避：减小 w_macro，或提高 every_n_steps。

2. 风险：图任务计算开销大，训练变慢。
   - 规避：降低辅助任务频率，先做 warm start。

3. 风险：数据接口不一致。
   - 规避：先固定主任务 DataLoader 与图数据构造，再逐步并行化。

## 9. 下一步建议

1. 将多任务脚本先稳定在单 fold 可复现。
2. 之后再接入 nested_cv.py，形成可发表级别评估闭环。
3. 最后再扩展第三任务（例如微观-宏观对齐）以进一步提升泛化。

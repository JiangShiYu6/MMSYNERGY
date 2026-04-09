# nested_cv.py 文件说明

本文档用于快速说明 nested_cv.py 的输入、输出，以及核心代码逻辑。

## 1. 这个文件是做什么的

nested_cv.py 用于执行嵌套交叉验证（Nested Cross Validation, NCV），目标是：

1. 在内层交叉验证中选择较优超参数组合。
2. 在外层测试折上评估该超参数组合的泛化表现。
3. 支持多进程 + 多 GPU 并行执行。

当前模型主体是 SynergyBert，损失是回归损失（MSE）。

## 2. 输入是什么

## 2.1 命令行输入

运行方式：

- python nested_cv.py <config_path>
- 可选覆盖参数：-u path.to.key=value

示例含义：

1. 第一个位置参数是配置文件路径。
2. -u 用于覆盖配置中的任意字段，例如学习率、fold、输出目录等。

## 2.2 配置文件输入

该脚本要求配置中至少包含以下关键字段：

1. task
   - 必须是 nested_cross_validation。

2. task_params
   - candidate_hps：超参数候选集合。
   - gpus：可用 GPU 列表。
   - folds：外层 fold 列表。

3. dataset
   - samples：样本表路径（包含药物对、细胞系、fold、标签）。
   - cell_protein_association：细胞-蛋白关联表路径。
   - train/valid/test 的 DataLoader 参数。

4. model
   - SynergyBert 所需结构参数。
   - 药物/蛋白特征文件路径。

5. trainer
   - epoch、optimizer、scheduler、patience 等训练参数。

6. model_dir
   - 输出目录。

## 2.3 数据层输入（由配置间接指定）

通过 SynergyDataset 读取：

1. 样本数据（drug_row_idx, drug_col_idx, cell_line_idx, fold, synergy 标签）。
2. 细胞对应蛋白和权重数据。

## 3. 输出是什么

脚本没有显式 return 文件给调用方，但会在 model_dir 及日志路径生成结果文件。

## 3.1 配置与过程输出

1. config_nested_cv.yml
   - 保存本次运行时实际配置。

2. hyperparam_combs.json
   - 展开后的超参数组合列表。

3. nested_cv.log
   - 多进程训练与结果日志。

## 3.2 预测结果输出

每个外层测试折会保存：

1. predictions-{test_fold}.csv
   - 包含该测试折样本及预测值。

## 4. 核心代码逻辑（简版）

可以按 6 步理解主流程：

1. 读取并补全配置
   - get_default_config 先设置默认 scheduler，再加载 yaml。

2. 构建数据加载器
   - get_dataloader 根据 train/valid/test fold 构造 DataLoader。

3. 单折训练评估
   - run_fold 负责一组 fold 配置下的训练、验证、测试。
   - 模型为 SynergyBert，优化器 AdamW，损失 MSE。

4. 内层并行超参评估
   - run_inner_task 在内层折上评估各超参数组合。
   - 结果写入 inner_result_queue。

5. 选择最优超参并执行外层测试
   - 每个外层 test_fold 收齐内层结果后，选择最优超参。
   - run_outer_fold 在该 test_fold 上训练并保存预测。

6. 汇总日志并结束
   - 主进程收集 outer_result_queue 结果并写日志。

## 5. 关键函数与职责

1. get_default_config
   - 读取配置并填默认值。

2. get_dataloader
   - 按 fold 构建 train/valid/test 数据加载器。

3. run_fold
   - 执行一次训练-验证-测试循环，返回最优验证与测试损失。

4. get_ncv_hps
   - 将 candidate_hps 展开为网格组合。

5. run_inner_task / run_outer_fold
   - 内层调参与外层评估的并行任务函数。

6. main
   - NCV 总调度入口（多进程与队列管理）。

## 6. 你阅读这个文件时建议先看哪里

1. main：先看并行调度与总体流程。
2. run_fold：再看单折训练逻辑。
3. get_dataloader：最后看数据如何按 fold 切分。

这样最快抓住这个脚本的主干。

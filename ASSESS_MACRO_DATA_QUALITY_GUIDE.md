# assess_macro_data_quality.py 文档

## 1. 这个文件是做什么的

[assess_macro_data_quality.py](assess_macro_data_quality.py) 是一个宏观异构图数据质量评估脚本。

核心思想是做扰动鲁棒性测试：

1. 在原始宏观图上随机删边和或删点。
2. 在扰动后的图上训练并评估链路预测。
3. 对比不同删除比例下的 AUC 和 AP 变化。

如果数据和图结构质量较高，通常会看到：

1. 删除比例越高，指标整体越容易下降。
2. 曲线整体更平滑，方差更可控。

## 2. 输入与输出

## 2.1 输入

脚本通过命令行参数和配置文件输入。

主要参数：

1. --config：宏观图配置文件，默认是 configs/macro.yml。
2. --gpu：GPU 编号，-1 表示 CPU。
3. --seed：随机种子基值。
4. --repeats：每个扰动设置重复次数。
5. --edge-rates：删边比例列表，逗号分隔。
6. --node-rates：删点比例列表，逗号分隔。
7. --epochs：每个设置最大训练轮数。
8. --patience：验证集 AUC 早停耐心值。
9. --lr：可选，覆盖配置里的学习率。
10. --weight-decay：可选，覆盖配置里的权重衰减。
11. --out-dir：结果输出目录。

配置文件依赖：

1. dataset：供 MacroNetDataset 读取图数据。
2. model：供 MacroEncoder 构建模型。
3. trainer.optimizer.lr 和 trainer.optimizer.weight_decay：在未手动覆盖时使用。

## 2.2 输出

输出文件保存在 --out-dir。

1. macro_quality_raw.csv：每次运行的原始结果。
2. macro_quality_summary.csv：按删除比例聚合后的均值和方差统计。

字段说明：

1. test_auc：测试集 ROC-AUC。
2. test_ap：测试集 Average Precision。
3. best_val_auc：训练过程中最佳验证 AUC。
4. n_nodes 和 n_edges：扰动后图规模。
5. n_pred_etypes：参与预测的边类型数量。

## 3. 主要函数说明

1. parse_rates
   作用：解析删边删点比例字符串并做范围检查。

2. load_base_graph
   作用：通过 MacroNetDataset 加载图和预测边类型。

3. random_remove_edges
   作用：按比例随机删除每种关系边。
   说明：删除 ID 的类型使用图的 idtype，避免 DGL 的 dtype 报错。

4. random_remove_nodes
   作用：按比例随机删除每种节点。
   说明：同样使用图 idtype。

5. split_edges
   作用：对每种目标边做 train val test 划分。

6. sample_negative_edges
   作用：为评估阶段按正样本数量采样负边。

7. evaluate_link_prediction
   作用：在指定 split 上计算加权 AUC 和 AP。

8. train_one_setting
   作用：对单个删边删点组合完成训练和评估。
   流程：扰动图 -> 划分边 -> 训练 -> 验证早停 -> 测试。

9. summarize_results
   作用：将原始结果按 edge_drop_rate 和 node_drop_rate 聚合。

10. main
    作用：命令行入口，负责遍历所有设置并保存结果。

## 4. 运行示例

快速对照：

python assess_macro_data_quality.py --config configs/macro.yml --gpu 0 --repeats 1 --edge-rates 0.0,0.1 --node-rates 0.0 --epochs 3 --patience 2 --out-dir output/quality_quick

正式实验建议：

python assess_macro_data_quality.py --config configs/macro.yml --gpu 0 --repeats 5 --edge-rates 0.0,0.05,0.1,0.2,0.3 --node-rates 0.0 --epochs 30 --patience 5 --out-dir output/quality_edge_curve

删点实验建议：

python assess_macro_data_quality.py --config configs/macro.yml --gpu 0 --repeats 5 --edge-rates 0.0 --node-rates 0.0,0.02,0.05,0.1 --epochs 30 --patience 5 --out-dir output/quality_node_curve

## 5. 如何解读结果

看 [macro_quality_summary.csv](output/quality_quick/macro_quality_summary.csv) 时建议关注：

1. 趋势：随着删除比例增加，auc_mean 和 ap_mean 是否整体下降。
2. 稳定性：auc_std 和 ap_std 是否较小。
3. 规模变化：n_edges 和 n_nodes 是否符合预期删除比例。

注意事项：

1. 小轮次和单次重复可能出现删边后指标暂时升高，这不代表数据质量更差或更好。
2. 结论应基于更多 repeats 和更充分训练后再判断。
3. 不同随机种子会有波动，建议固定 seed 后再做多种子复核。

## 6. 当前实现边界

1. 这是链路预测鲁棒性评估，不直接评估最终协同预测主任务。
2. 负采样采用随机采样并排除正边，不是全图精确枚举。
3. 当某个边类型样本过少时，会被自动跳过。

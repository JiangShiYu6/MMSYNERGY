# infer_macro.py 文档

## 1. 这个文件的任务是什么

infer_macro.py 的任务是：

1. 加载已经训练好的 MacroEncoder 权重（pretrain_model_path）。
2. 加载宏观异构图数据（drug/protein/sideeffect）。
3. 对整张图做一次前向推理。
4. 将节点表示中的 drug embedding 和 protein embedding 分别保存到文件。

简单说：它不是训练脚本，而是“用已训练宏观编码器导出宏观特征”的推理脚本。

## 2. 输入与输出

## 2.1 输入

1. 命令行输入
   - config（必填）：配置文件路径
   - -u / --update（可选）：覆盖配置，格式 path.to.key=value

2. 配置输入（关键字段）
   - dataset：构图数据路径和参数（传给 MacroNetDataset）
   - model：MacroEncoder 结构参数（num_layers、hidden_dim、num_heads、dropout）
   - pretrain_model_path：训练好的模型 ckpt 路径
   - save_path.drug：drug embedding 输出路径
   - save_path.protein：protein embedding 输出路径
   - gpu：设备编号

## 2.2 输出

1. save_path.drug
   - torch 保存的 drug 节点 embedding 张量（.pth）

2. save_path.protein
   - torch 保存的 protein 节点 embedding 张量（.pth）

3. infer.log
   - 推理日志（数据构建、模型加载、推理完成）

## 3. 每个函数有什么用

## get_dataset(config, device)

作用：

1. 实例化 MacroNetDataset。
2. 调用 dataset.load() 加载图缓存。
3. 取出图对象 macro_graph 和边信息 edge_splits。
4. 额外遍历各关系边，整理 all_edges（当前主流程未使用）。
5. 将图放到目标设备并返回。

返回：

1. macro_graph：DGL 异构图（在 CPU/GPU 上）
2. all_edges：每种边类型的全量边索引（当前作为辅助信息）

说明：

- 代码里变量名写成 edge_splits，但 get_dataset 最终返回的是 all_edges。
- 在 main 中这个返回值目前也没有被后续使用，不影响导出 embedding。

## main(config)

作用：

1. 创建输出目录（drug/protein 两个保存路径的父目录）。
2. 创建日志器 infer.log。
3. 选择设备（gpu<0 或无 CUDA 时使用 CPU）。
4. 加载宏观图数据。
5. 构建 MacroEncoder（输入维度从图节点特征自动读取）。
6. 加载预训练权重并进入 eval 模式。
7. 前向推理得到各节点类型 embedding。
8. 将 drug/protein embedding 分别保存为 .pth。

## **main** 入口

作用：

1. 解析命令行参数。
2. 从配置文件加载 BaseConfig。
3. 应用 -u 覆盖参数。
4. 调用 main(config)。

## 4. 代码逻辑（简版流程）

1. 读配置并确定输出目录。
2. 载入宏观图（节点与关系）。
3. 用图中节点特征维度初始化 MacroEncoder。
4. 加载训练好的 ckpt。
5. 对整张图前向推理。
6. 保存 drug/protein 两类节点向量。

## 5. 用具体数值举个例子

假设你的图数据统计如下：

1. drug 节点数：38
2. protein 节点数：4871
3. sideeffect 节点数：120
4. 节点输入特征维度：
   - drug: 256
   - protein: 256
   - sideeffect: 256

配置里设定：

1. hidden_dim = 128
2. num_layers = 2
3. num_heads = 8

推理时会发生：

1. MacroEncoder 输入 in_dims = {drug:256, protein:256, sideeffect:256}
2. 前向后得到：
   - model_output['drug'] 形状约为 [38, 128]
   - model_output['protein'] 形状约为 [4871, 128]
   - model_output['sideeffect'] 形状约为 [120, 128]
3. 脚本只保存前两者：
   - output/fuse/drug_macro_feat.pth（38 x 128）
   - output/fuse/protein_macro_feat.pth（4871 x 128）

这两个文件通常会被后续融合或下游预测阶段直接加载使用。

## 6. 命令行示例

```bash
python infer_macro.py configs/macro_infer.yml -u "gpu=0" "pretrain_model_path=output/pretrain_macro/macro_encoder/best.ckpt"
```

含义：

1. 使用 macro_infer 配置执行推理。
2. 指定在 GPU0 推理。
3. 指定宏观编码器权重路径。

## 7. 两个注意点

1. 这是推理脚本，不会更新模型参数。
2. get_dataset 返回的第二个变量目前未被 main 使用，后续可清理或用于额外评估逻辑。

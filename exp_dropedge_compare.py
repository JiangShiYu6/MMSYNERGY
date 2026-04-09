"""
DropEdge 对比实验：区分正则化效应 vs 噪声效应

三组实验：
  A. Baseline：完整图训练，完整图评估
  B. DropEdge：每 epoch 随机丢 drop_rate 的边训练，完整图评估
  C. 固定删边：训练前固定删 drop_rate 的边，在删减后的图上训练和评估

判断标准：
  - B >> A, C ≈ A → 正则化效应为主
  - C >> A, B ≈ A → 噪声效应为主
  - B > A 且 C > A → 两者兼有

Usage:
    python exp_dropedge_compare.py --config configs/macro.yml --drop-rate 0.2 --repeats 5 --epochs 30
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW

from my_config import BaseConfig
from models.datasets import MacroNetDataset
from models.models import MacroEncoder
from models.utils import seet_random_seed


# ============================================================
#  Graph utilities
# ============================================================

def load_base_graph(config: BaseConfig):
    ds_cfg = config.dataset
    dataset = MacroNetDataset(**ds_cfg)
    try:
        dataset.load()
        g, _ = dataset[0]
    except Exception:
        dataset.process()
        g, _ = dataset[0]
    return g, dataset.pred_edge_types, float(ds_cfg.val_rate), float(ds_cfg.test_rate)


def split_edges(g, pred_edge_types, val_rate, test_rate, rng, min_edges=30):
    """Split edges into train/val/test for link prediction evaluation."""
    edge_splits = {}
    for etype in pred_edge_types:
        if etype not in g.etypes:
            continue
        u, v = g.edges(etype=etype)
        n = u.shape[0]
        if n < min_edges:
            continue
        perm = rng.permutation(n)
        n_test = max(1, int(n * test_rate))
        n_val = max(1, int(n * val_rate))
        n_train = n - n_val - n_test
        if n_train < 1:
            continue
        edge_splits[etype] = {
            "train": [u[perm[:n_train]], v[perm[:n_train]]],
            "val": [u[perm[n_train:n_train + n_val]], v[perm[n_train:n_train + n_val]]],
            "test": [u[perm[n_train + n_val:]], v[perm[n_train + n_val:]]],
        }
    return edge_splits


def random_remove_edges(g, drop_rate, rng):
    """Permanently remove a fraction of edges from the graph."""
    if drop_rate <= 0.0:
        return g
    for canonical_etype in g.canonical_etypes:
        n_edges = g.num_edges(canonical_etype)
        if n_edges <= 1:
            continue
        n_drop = min(int(n_edges * drop_rate), n_edges - 1)
        if n_drop <= 0:
            continue
        drop_eids = rng.choice(n_edges, size=n_drop, replace=False)
        drop_eids = torch.as_tensor(drop_eids, dtype=g.idtype)
        g = dgl.remove_edges(g, drop_eids, etype=canonical_etype)
    return g


def dropedge_epoch(g, drop_rate, rng):
    """
    Create a temporary copy of the graph with randomly dropped edges.
    Used during training only; evaluation uses the full graph.
    """
    if drop_rate <= 0.0:
        return g

    edge_dict = {}
    for canonical_etype in g.canonical_etypes:
        src, dst = g.edges(etype=canonical_etype)
        n_edges = src.shape[0]
        if n_edges <= 1:
            edge_dict[canonical_etype] = (src, dst)
            continue
        n_keep = max(1, n_edges - int(n_edges * drop_rate))
        keep_idx = rng.choice(n_edges, size=n_keep, replace=False)
        keep_idx = torch.as_tensor(keep_idx, dtype=torch.long)
        edge_dict[canonical_etype] = (src[keep_idx], dst[keep_idx])

    # Build new graph with same node counts
    num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    g_dropped = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict, idtype=g.idtype)

    # Copy node features
    for ntype in g.ntypes:
        for key, val in g.nodes[ntype].data.items():
            g_dropped.nodes[ntype].data[key] = val

    return g_dropped


# ============================================================
#  Evaluation
# ============================================================

def sample_negative_edges(pos_u, pos_v, num_src, num_dst, n_samples, rng):
    if n_samples <= 0:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
    pos_set = set(zip(pos_u.cpu().numpy().tolist(), pos_v.cpu().numpy().tolist()))
    neg_pairs = set()
    max_trials = max(1000, n_samples * 30)
    trials = 0
    while len(neg_pairs) < n_samples and trials < max_trials:
        batch = min(4 * (n_samples - len(neg_pairs)), 8192)
        cand_u = rng.integers(0, num_src, size=batch)
        cand_v = rng.integers(0, num_dst, size=batch)
        for uu, vv in zip(cand_u.tolist(), cand_v.tolist()):
            if (uu, vv) not in pos_set:
                neg_pairs.add((uu, vv))
            if len(neg_pairs) >= n_samples:
                break
        trials += batch
    if len(neg_pairs) == 0:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
    neg_u, neg_v = zip(*neg_pairs)
    return torch.as_tensor(neg_u, dtype=torch.int64), torch.as_tensor(neg_v, dtype=torch.int64)


def evaluate_link_prediction(model, g, edge_splits, split, rng):
    """Evaluate link prediction AUC and AP on given split."""
    model.eval()
    auc_w, ap_w, n_total = 0.0, 0.0, 0
    with torch.no_grad():
        h = model(g)
        for etype, splits in edge_splits.items():
            pos_u, pos_v = splits[split]
            n_pos = pos_u.shape[0]
            if n_pos < 2:
                continue
            src_ntype, dst_ntype = etype.split("2")
            neg_u, neg_v = sample_negative_edges(
                pos_u, pos_v, h[src_ntype].shape[0], h[dst_ntype].shape[0], n_pos, rng
            )
            if neg_u.shape[0] < 2:
                continue
            # Score positive edges
            pos_feat = torch.cat([h[src_ntype][pos_u], h[dst_ntype][pos_v]], dim=1)
            pos_scores = model.link_pred_heads[etype](pos_feat).flatten().cpu().numpy()
            # Score negative edges
            neg_feat = torch.cat([h[src_ntype][neg_u.to(pos_u.device)],
                                  h[dst_ntype][neg_v.to(pos_v.device)]], dim=1)
            neg_scores = model.link_pred_heads[etype](neg_feat).flatten().cpu().numpy()

            y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
            y_score = np.concatenate([pos_scores, neg_scores])
            try:
                auc = roc_auc_score(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
            except Exception:
                continue
            auc_w += auc * n_pos
            ap_w += ap * n_pos
            n_total += n_pos
    if n_total == 0:
        return float("nan"), float("nan")
    return auc_w / n_total, ap_w / n_total


# ============================================================
#  Training one setting
# ============================================================

@dataclass
class ExpResult:
    method: str        # "baseline", "dropedge", "fixed_remove"
    drop_rate: float
    repeat_id: int
    best_val_auc: float
    test_auc: float
    test_ap: float


def train_one_setting(
    g_full,                 # full graph (never modified)
    pred_edge_types,
    model_cfg,
    val_rate,
    test_rate,
    method,                 # "baseline", "dropedge", "fixed_remove"
    drop_rate,
    repeat_seed,
    device,
    epochs,
    patience,
    lr,
    weight_decay,
):
    rng = np.random.default_rng(repeat_seed)

    # --- Prepare graphs depending on method ---
    if method == "fixed_remove":
        # Fixed removal: permanently remove edges, train & evaluate on reduced graph
        g_train_base = random_remove_edges(g_full.clone(), drop_rate, rng)
        g_eval = g_train_base  # evaluate on the same reduced graph
    else:
        # Baseline and DropEdge: always evaluate on full graph
        g_train_base = g_full
        g_eval = g_full

    # --- Split edges (from eval graph) for val/test ---
    edge_splits = split_edges(g_eval, pred_edge_types, val_rate, test_rate, rng)
    if len(edge_splits) == 0:
        return ExpResult(method, drop_rate, repeat_seed,
                         float("nan"), float("nan"), float("nan"))

    g_eval = g_eval.to(device)
    g_train_base = g_train_base.to(device)
    edge_splits = {
        etype: {split: [uv[0].to(device), uv[1].to(device)] for split, uv in sp.items()}
        for etype, sp in edge_splits.items()
    }

    # --- Build model ---
    in_dims = {nt: g_eval.nodes[nt].data["feat"].shape[1] for nt in g_eval.ntypes}
    model = MacroEncoder(in_dims=in_dims, **model_cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state, best_val_auc, bad_epochs = None, -1.0, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # --- Select training graph for this epoch ---
        if method == "dropedge":
            # DropEdge: randomly drop edges THIS epoch only
            epoch_rng = np.random.default_rng(repeat_seed * 10000 + epoch)
            g_train = dropedge_epoch(g_train_base, drop_rate, epoch_rng).to(device)
        else:
            # Baseline or fixed_remove: use the same graph every epoch
            g_train = g_train_base

        # --- Forward + loss on training graph ---
        h = model(g_train)
        losses = []
        for etype in edge_splits.keys():
            losses.append(model.link_pred_loss(h, edge_splits, etype, split="train"))
        if len(losses) == 0:
            break
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

        # --- Evaluate on eval graph (full graph for baseline/dropedge, reduced for fixed) ---
        val_auc, _ = evaluate_link_prediction(model, g_eval, edge_splits, "val", rng)
        if np.isnan(val_auc):
            continue
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_auc, test_ap = evaluate_link_prediction(model, g_eval, edge_splits, "test", rng)
    return ExpResult(
        method=method,
        drop_rate=drop_rate,
        repeat_id=repeat_seed,
        best_val_auc=best_val_auc,
        test_auc=test_auc,
        test_ap=test_ap,
    )


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("DropEdge comparison: baseline vs dropedge vs fixed removal")
    parser.add_argument("--config", type=str, default="configs/macro.yml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--repeats", type=int, default=5, help="number of repeats per method")
    parser.add_argument("--drop-rate", type=float, default=0.2, help="edge drop rate")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default="output/dropedge_compare")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    config = BaseConfig()
    config.load_from_file(args.config)
    seet_random_seed(args.seed)

    device = "cpu"
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.gpu}"

    g_base, pred_edge_types, val_rate, test_rate = load_base_graph(config)
    lr = float(args.lr if args.lr is not None else config.trainer.optimizer.lr)
    wd = float(args.weight_decay if args.weight_decay is not None else config.trainer.optimizer.weight_decay)

    methods = ["baseline", "dropedge", "fixed_remove"]
    total_runs = len(methods) * args.repeats
    all_results: List[ExpResult] = []
    run_idx = 0

    for method in methods:
        for rep in range(args.repeats):
            run_idx += 1
            repeat_seed = args.seed + rep
            dr = args.drop_rate if method != "baseline" else 0.0
            print(f"[{run_idx:03d}/{total_runs:03d}] {method:>13s} | drop_rate={dr:.2f} | seed={repeat_seed}")
            res = train_one_setting(
                g_full=g_base,
                pred_edge_types=pred_edge_types,
                model_cfg=config.model,
                val_rate=val_rate,
                test_rate=test_rate,
                method=method,
                drop_rate=args.drop_rate,
                repeat_seed=repeat_seed,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                lr=lr,
                weight_decay=wd,
            )
            all_results.append(res)
            print(f"       val_auc={res.best_val_auc:.4f}  test_auc={res.test_auc:.4f}  test_ap={res.test_ap:.4f}")

    # --- Summary ---
    raw_df = pd.DataFrame([r.__dict__ for r in all_results])
    grp = raw_df.groupby("method", as_index=False)
    summary = grp.agg(
        auc_mean=("test_auc", "mean"),
        auc_std=("test_auc", "std"),
        ap_mean=("test_ap", "mean"),
        ap_std=("test_ap", "std"),
        val_auc_mean=("best_val_auc", "mean"),
        runs=("repeat_id", "count"),
    )
    # Reorder for readability
    method_order = {"baseline": 0, "dropedge": 1, "fixed_remove": 2}
    summary["_order"] = summary["method"].map(method_order)
    summary = summary.sort_values("_order").drop(columns="_order")

    raw_fp = os.path.join(args.out_dir, "dropedge_compare_raw.csv")
    summary_fp = os.path.join(args.out_dir, "dropedge_compare_summary.csv")
    raw_df.to_csv(raw_fp, index=False)
    summary.to_csv(summary_fp, index=False)

    print("\n" + "=" * 70)
    print(f"  DropEdge Comparison (drop_rate={args.drop_rate})")
    print("=" * 70)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # --- Interpretation ---
    baseline_auc = summary.loc[summary["method"] == "baseline", "auc_mean"].values[0]
    dropedge_auc = summary.loc[summary["method"] == "dropedge", "auc_mean"].values[0]
    fixed_auc = summary.loc[summary["method"] == "fixed_remove", "auc_mean"].values[0]

    print("Interpretation:")
    de_diff = dropedge_auc - baseline_auc
    fr_diff = fixed_auc - baseline_auc
    threshold = 0.005  # 0.5% as significance threshold

    if de_diff > threshold and fr_diff <= threshold:
        print(f"  DropEdge AUC +{de_diff:.4f}, Fixed +{fr_diff:.4f}")
        print("  → Regularization effect dominates.")
        print("  → Model overfits to graph structure; DropEdge suffices.")
    elif fr_diff > threshold and de_diff <= threshold:
        print(f"  DropEdge AUC +{de_diff:.4f}, Fixed +{fr_diff:.4f}")
        print("  → Noise effect dominates.")
        print("  → Graph contains harmful noisy edges; edge reconstruction recommended.")
    elif de_diff > threshold and fr_diff > threshold:
        print(f"  DropEdge AUC +{de_diff:.4f}, Fixed +{fr_diff:.4f}")
        print("  → Both regularization and noise effects present.")
        print("  → Consider combining DropEdge with edge reconstruction.")
    else:
        print(f"  DropEdge AUC +{de_diff:.4f}, Fixed +{fr_diff:.4f}")
        print("  → Neither method shows significant improvement.")
        print("  → The baseline may already be well-regularized at this drop rate.")

    print(f"\nSaved raw results to: {raw_fp}")
    print(f"Saved summary to: {summary_fp}")


if __name__ == "__main__":
    main()

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import dgl
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW

from my_config import BaseConfig
from models.datasets import MacroNetDataset
from models.models import MacroEncoder
from models.utils import seet_random_seed


@dataclass
class EvalResult:
    edge_drop_rate: float
    node_drop_rate: float
    repeat_id: int
    best_val_auc: float
    test_auc: float
    test_ap: float
    n_nodes: int
    n_edges: int
    n_pred_etypes: int


def parse_rates(rate_str: str) -> List[float]:
    rates = []
    for x in rate_str.split(","):
        v = float(x.strip())
        if v < 0.0 or v >= 1.0:
            raise ValueError(f"drop rates must be in [0, 1), found {v}")
        rates.append(v)
    return rates


def choose_device(gpu_id: int) -> str:
    if gpu_id < 0 or not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{gpu_id}"


def load_base_graph(config: BaseConfig) -> Tuple[dgl.DGLHeteroGraph, List[str], float, float]:
    ds_cfg = config.dataset
    dataset = MacroNetDataset(**ds_cfg)

    try:
        dataset.load()
        g, _ = dataset[0]
    except Exception:
        dataset.process()
        g, _ = dataset[0]

    val_rate = float(ds_cfg.val_rate)
    test_rate = float(ds_cfg.test_rate)
    return g, dataset.pred_edge_types, val_rate, test_rate


def random_remove_edges(
    g: dgl.DGLHeteroGraph,
    drop_rate: float,
    rng: np.random.Generator,
) -> dgl.DGLHeteroGraph:
    if drop_rate <= 0.0:
        return g

    g_new = g
    for canonical_etype in g_new.canonical_etypes:
        n_edges = g_new.num_edges(canonical_etype)
        if n_edges <= 1:
            continue
        n_drop = int(n_edges * drop_rate)
        if n_drop <= 0:
            continue
        n_drop = min(n_drop, n_edges - 1)
        drop_eids = rng.choice(n_edges, size=n_drop, replace=False)
        drop_eids = torch.as_tensor(drop_eids, dtype=g_new.idtype)
        g_new = dgl.remove_edges(g_new, drop_eids, etype=canonical_etype)
    return g_new


def random_remove_nodes(
    g: dgl.DGLHeteroGraph,
    drop_rate: float,
    rng: np.random.Generator,
) -> dgl.DGLHeteroGraph:
    if drop_rate <= 0.0:
        return g

    g_new = g
    for ntype in g_new.ntypes:
        n_nodes = g_new.num_nodes(ntype)
        if n_nodes <= 2:
            continue
        n_drop = int(n_nodes * drop_rate)
        if n_drop <= 0:
            continue
        n_drop = min(n_drop, n_nodes - 2)
        drop_nids = rng.choice(n_nodes, size=n_drop, replace=False)
        drop_nids = torch.as_tensor(drop_nids, dtype=g_new.idtype)
        g_new = dgl.remove_nodes(g_new, drop_nids, ntype=ntype)
    return g_new


def split_edges(
    g: dgl.DGLHeteroGraph,
    pred_edge_types: List[str],
    val_rate: float,
    test_rate: float,
    rng: np.random.Generator,
    min_edges_per_type: int = 30,
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    edge_splits: Dict[str, Dict[str, List[torch.Tensor]]] = {}

    for etype in pred_edge_types:
        if etype not in g.etypes:
            continue

        u, v = g.edges(etype=etype)
        n = u.shape[0]
        if n < min_edges_per_type:
            continue

        perm = rng.permutation(n)
        n_test = max(1, int(n * test_rate))
        n_val = max(1, int(n * val_rate))
        n_train = n - n_val - n_test
        if n_train < 1:
            continue

        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]

        edge_splits[etype] = {
            "train": [u[train_idx], v[train_idx]],
            "val": [u[val_idx], v[val_idx]],
            "test": [u[test_idx], v[test_idx]],
        }

    return edge_splits


def sample_negative_edges(
    pos_u: torch.Tensor,
    pos_v: torch.Tensor,
    num_src: int,
    num_dst: int,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def score_edges(
    model: MacroEncoder,
    h: Dict[str, torch.Tensor],
    etype: str,
    u: torch.Tensor,
    v: torch.Tensor,
) -> np.ndarray:
    src_ntype, dst_ntype = etype.split("2")
    uv = torch.cat([h[src_ntype][u], h[dst_ntype][v]], dim=1)
    scores = model.link_pred_heads[etype](uv).detach().flatten().cpu().numpy()
    return scores


def evaluate_link_prediction(
    model: MacroEncoder,
    g: dgl.DGLHeteroGraph,
    edge_splits: Dict[str, Dict[str, List[torch.Tensor]]],
    split: str,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    model.eval()
    auc_weighted = 0.0
    ap_weighted = 0.0
    n_total_pos = 0

    with torch.no_grad():
        h = model(g)
        for etype, splits in edge_splits.items():
            pos_u, pos_v = splits[split]
            n_pos = pos_u.shape[0]
            if n_pos < 2:
                continue

            src_ntype, dst_ntype = etype.split("2")
            neg_u, neg_v = sample_negative_edges(
                pos_u,
                pos_v,
                num_src=h[src_ntype].shape[0],
                num_dst=h[dst_ntype].shape[0],
                n_samples=n_pos,
                rng=rng,
            )
            if neg_u.shape[0] < 2:
                continue

            pos_scores = score_edges(model, h, etype, pos_u, pos_v)
            neg_scores = score_edges(model, h, etype, neg_u.to(pos_u.device), neg_v.to(pos_v.device))
            y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
            y_score = np.concatenate([pos_scores, neg_scores])

            try:
                auc = roc_auc_score(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
            except Exception:
                continue

            auc_weighted += auc * n_pos
            ap_weighted += ap * n_pos
            n_total_pos += n_pos

    if n_total_pos == 0:
        return float("nan"), float("nan")

    return auc_weighted / n_total_pos, ap_weighted / n_total_pos


def train_one_setting(
    g_base: dgl.DGLHeteroGraph,
    pred_edge_types: List[str],
    model_cfg: BaseConfig,
    val_rate: float,
    test_rate: float,
    edge_drop_rate: float,
    node_drop_rate: float,
    repeat_seed: int,
    device: str,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> EvalResult:
    rng = np.random.default_rng(repeat_seed)

    g = g_base
    g = random_remove_edges(g, edge_drop_rate, rng)
    g = random_remove_nodes(g, node_drop_rate, rng)

    edge_splits = split_edges(g, pred_edge_types, val_rate, test_rate, rng)
    if len(edge_splits) == 0:
        return EvalResult(
            edge_drop_rate=edge_drop_rate,
            node_drop_rate=node_drop_rate,
            repeat_id=repeat_seed,
            best_val_auc=float("nan"),
            test_auc=float("nan"),
            test_ap=float("nan"),
            n_nodes=sum(g.num_nodes(nt) for nt in g.ntypes),
            n_edges=sum(g.num_edges(et) for et in g.etypes),
            n_pred_etypes=0,
        )

    g = g.to(device)
    edge_splits = {
        etype: {
            split: [uv[0].to(device), uv[1].to(device)]
            for split, uv in splits.items()
        }
        for etype, splits in edge_splits.items()
    }

    model = MacroEncoder(
        in_dims={nt: g.nodes[nt].data["feat"].shape[1] for nt in g.ntypes},
        **model_cfg,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_auc = -1.0
    bad_epochs = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        h = model(g)

        losses = []
        for etype in edge_splits.keys():
            losses.append(model.link_pred_loss(h, edge_splits, etype, split="train"))

        if len(losses) == 0:
            break

        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

        val_auc, _ = evaluate_link_prediction(model, g, edge_splits, split="val", rng=rng)
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

    test_auc, test_ap = evaluate_link_prediction(model, g, edge_splits, split="test", rng=rng)
    return EvalResult(
        edge_drop_rate=edge_drop_rate,
        node_drop_rate=node_drop_rate,
        repeat_id=repeat_seed,
        best_val_auc=best_val_auc,
        test_auc=test_auc,
        test_ap=test_ap,
        n_nodes=sum(g.num_nodes(nt) for nt in g.ntypes),
        n_edges=sum(g.num_edges(et) for et in g.etypes),
        n_pred_etypes=len(edge_splits),
    )


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["edge_drop_rate", "node_drop_rate"], as_index=False)
    summary = grp.agg(
        auc_mean=("test_auc", "mean"),
        auc_std=("test_auc", "std"),
        ap_mean=("test_ap", "mean"),
        ap_std=("test_ap", "std"),
        val_auc_mean=("best_val_auc", "mean"),
        runs=("repeat_id", "count"),
        n_nodes=("n_nodes", "mean"),
        n_edges=("n_edges", "mean"),
        n_pred_etypes=("n_pred_etypes", "mean"),
    )
    return summary.sort_values(["edge_drop_rate", "node_drop_rate"])


def main():
    parser = argparse.ArgumentParser("Assess macro graph data quality via perturbation + link prediction")
    parser.add_argument("--config", type=str, default="configs/macro.yml", help="path to macro config")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id, use -1 for cpu")
    parser.add_argument("--seed", type=int, default=18, help="base random seed")
    parser.add_argument("--repeats", type=int, default=3, help="repeats for each perturbation setting")
    parser.add_argument("--edge-rates", type=str, default="0.0,0.1,0.2,0.3", help="comma-separated edge drop rates")
    parser.add_argument("--node-rates", type=str, default="0.0", help="comma-separated node drop rates")
    parser.add_argument("--epochs", type=int, default=30, help="max training epochs per setting")
    parser.add_argument("--patience", type=int, default=5, help="early-stop patience on validation AUC")
    parser.add_argument("--lr", type=float, default=None, help="override learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="override weight decay")
    parser.add_argument("--out-dir", type=str, default="output/data_quality", help="output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    config = BaseConfig()
    config.load_from_file(args.config)
    seet_random_seed(args.seed)

    edge_rates = parse_rates(args.edge_rates)
    node_rates = parse_rates(args.node_rates)
    device = choose_device(args.gpu)

    g_base, pred_edge_types, val_rate, test_rate = load_base_graph(config)

    lr = float(args.lr if args.lr is not None else config.trainer.optimizer.lr)
    wd = float(args.weight_decay if args.weight_decay is not None else config.trainer.optimizer.weight_decay)

    all_results: List[EvalResult] = []
    total_runs = len(edge_rates) * len(node_rates) * args.repeats
    run_idx = 0

    for er in edge_rates:
        for nr in node_rates:
            for rep in range(args.repeats):
                run_idx += 1
                repeat_seed = args.seed + rep
                print(
                    f"[Run {run_idx:03d}/{total_runs:03d}] edge_drop={er:.2f}, node_drop={nr:.2f}, seed={repeat_seed}"
                )
                res = train_one_setting(
                    g_base=g_base,
                    pred_edge_types=pred_edge_types,
                    model_cfg=config.model,
                    val_rate=val_rate,
                    test_rate=test_rate,
                    edge_drop_rate=er,
                    node_drop_rate=nr,
                    repeat_seed=repeat_seed,
                    device=device,
                    epochs=args.epochs,
                    patience=args.patience,
                    lr=lr,
                    weight_decay=wd,
                )
                all_results.append(res)
                print(
                    f"  -> val_auc={res.best_val_auc:.4f}, test_auc={res.test_auc:.4f}, test_ap={res.test_ap:.4f}, pred_etypes={res.n_pred_etypes}"
                )

    raw_df = pd.DataFrame([r.__dict__ for r in all_results])
    summary_df = summarize_results(raw_df)

    raw_fp = os.path.join(args.out_dir, "macro_quality_raw.csv")
    summary_fp = os.path.join(args.out_dir, "macro_quality_summary.csv")
    raw_df.to_csv(raw_fp, index=False)
    summary_df.to_csv(summary_fp, index=False)

    print("\n===== Quality Check Summary =====")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved raw results to: {raw_fp}")
    print(f"Saved summary to: {summary_fp}")


if __name__ == "__main__":
    main()

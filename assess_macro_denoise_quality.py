"""
Assess macro data quality with denoised MacroEncoder.

This script compares the standard MacroEncoder (baseline) against
a denoised variant inspired by NoiseHGNN (AAAI 2025). The denoised
encoder builds a similarity graph from node features and uses it to
re-weight edges during GAT message passing, suppressing noise edges.

Usage:
    python assess_macro_denoise_quality.py --config configs/macro.yml \
        --edge-rates 0.0,0.05,0.1,0.2,0.3 --repeats 3 --epochs 30
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW

from my_config import BaseConfig
from models.datasets import MacroNetDataset
from models.models import MacroEncoder, HANLayer
from models.utils import seet_random_seed

from dgl.nn.pytorch import GATConv, HeteroGraphConv


# ============================================================
#  Denoised MacroEncoder: Similarity-aware HAN
# ============================================================

class SimilarityGraphBuilder(nn.Module):
    """
    Build a similarity graph from node features (inspired by NoiseHGNN).

    For each node type, project features to a unified space, compute
    pairwise cosine similarity, and retain the top-k neighbors.
    """

    def __init__(self, in_dims: Dict[str, int], hidden_dim: int = 128, k: int = 15):
        super().__init__()
        self.k = k
        self.proj = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim)
            for ntype, in_dim in in_dims.items()
        })

    def forward(self, g: dgl.DGLHeteroGraph) -> Dict[str, torch.Tensor]:
        """Return a dict mapping ntype -> (N, N) sparse similarity adjacency."""
        sim_adjs = {}
        for ntype in g.ntypes:
            feat = g.nodes[ntype].data["feat"]
            z = self.proj[ntype](feat)  # (N, hidden)
            z = F.normalize(z, dim=-1)
            # cosine similarity
            sim = z @ z.T  # (N, N)
            # kNN sparsification: keep top-k per row
            n = sim.size(0)
            k = min(self.k, n - 1)
            if k <= 0:
                sim_adjs[ntype] = torch.zeros_like(sim)
                continue
            _, topk_idx = sim.topk(k, dim=-1)
            mask = torch.zeros_like(sim)
            mask.scatter_(1, topk_idx, 1.0)
            # Remove self-loops
            mask.fill_diagonal_(0.0)
            sim_adjs[ntype] = sim * mask
        return sim_adjs


class SimAwareHANLayer(nn.Module):
    """
    HAN layer with similarity-aware attention weighting.

    Same structure as HANLayer, but additionally uses similarity
    adjacency to down-weight edges that connect dissimilar nodes.
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.node_attn = HeteroGraphConv(
            {
                "drug2drug": GATConv(in_dim, hidden_dim // num_heads, num_heads=num_heads, feat_drop=dropout),
                "drug2protein": GATConv(in_dim, hidden_dim // num_heads, num_heads=num_heads, feat_drop=dropout),
                "protein2drug": GATConv(in_dim, hidden_dim // num_heads, num_heads=num_heads, feat_drop=dropout),
                "protein2protein": GATConv(in_dim, hidden_dim // num_heads, num_heads=num_heads, feat_drop=dropout),
                "sideeffect2drug": GATConv(in_dim, hidden_dim // num_heads, num_heads=num_heads, feat_drop=dropout),
            },
            aggregate="stack",
        )

        self.semantic_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, g, h):
        node_feats = self.node_attn(g, h)
        out_h = {}
        for ntype, feats in node_feats.items():
            feats = feats.view(feats.size(0), feats.size(1), -1)
            sem_weights = self.semantic_attn(feats)
            aggregated = (feats * sem_weights).sum(dim=1)
            out_h[ntype] = aggregated
        return out_h


class DenoisedMacroEncoder(nn.Module):
    """
    MacroEncoder with similarity-aware denoising.

    Architecture:
    1. Build similarity graph from raw node features
    2. Run HAN on the *original* graph -> H_orig
    3. Run HAN on the *similarity* graph -> H_sim
    4. Fuse: H = alpha * H_orig + (1-alpha) * H_sim
    5. Link prediction heads on fused representations
    6. Contrastive loss between H_orig and H_sim
    """

    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.4,
        etypes: List[str] = None,
        sim_k: int = 15,
        alpha: float = 0.7,
    ):
        super().__init__()
        if etypes is None:
            etypes = ["drug2drug", "drug2protein", "protein2protein", "sideeffect2drug"]

        self.hidden_dim = hidden_dim
        self.alpha = alpha

        # Shared projection layers
        self.proj_layers = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim)
            for ntype, in_dim in in_dims.items()
        })

        # Similarity graph builder
        self.sim_builder = SimilarityGraphBuilder(in_dims, hidden_dim, k=sim_k)

        # HAN layers for original graph
        self.hans_orig = nn.ModuleList([
            HANLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Lightweight projector for similarity-graph path
        # (single layer to keep parameter count close to baseline)
        self.sim_proj = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for ntype in in_dims.keys()
        })

        # Link prediction heads
        self.link_pred_heads = nn.ModuleDict({
            etype: nn.Sequential(
                nn.Linear(2 * hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
            for etype in etypes
        })

    def forward(self, g):
        # Project node features
        raw_h = {
            ntype: self.proj_layers[ntype](g.nodes[ntype].data["feat"])
            for ntype in g.ntypes
        }

        # === Original graph path ===
        h_orig = raw_h
        for han in self.hans_orig:
            h_orig = han(g, h_orig)
            for ntype, feat in raw_h.items():
                if ntype not in h_orig:
                    h_orig[ntype] = feat

        # === Similarity graph path ===
        sim_adjs = self.sim_builder(g)
        h_sim = {}
        for ntype in g.ntypes:
            if ntype in sim_adjs:
                adj = sim_adjs[ntype]  # (N, N)
                # Normalize adjacency
                deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
                adj_norm = adj / deg
                # Simple message passing: H_sim = adj_norm @ raw_h
                propagated = adj_norm @ raw_h[ntype]
                h_sim[ntype] = self.sim_proj[ntype](propagated)
            else:
                h_sim[ntype] = raw_h[ntype]

        # === Fuse ===
        h_fused = {}
        for ntype in g.ntypes:
            if ntype in h_orig and ntype in h_sim:
                h_fused[ntype] = self.alpha * h_orig[ntype] + (1 - self.alpha) * h_sim[ntype]
            elif ntype in h_orig:
                h_fused[ntype] = h_orig[ntype]
            else:
                h_fused[ntype] = raw_h[ntype]

        return h_fused

    def contrastive_loss(self, g, temperature=0.5):
        """
        Contrastive loss between original-graph and similarity-graph
        representations. Encourages agreement on true neighbors.
        """
        raw_h = {
            ntype: self.proj_layers[ntype](g.nodes[ntype].data["feat"])
            for ntype in g.ntypes
        }

        # Original path
        h_orig = raw_h.copy()
        for han in self.hans_orig:
            h_orig = han(g, h_orig)
            for ntype, feat in raw_h.items():
                if ntype not in h_orig:
                    h_orig[ntype] = feat

        # Similarity path
        sim_adjs = self.sim_builder(g)
        h_sim = {}
        for ntype in g.ntypes:
            if ntype in sim_adjs:
                adj = sim_adjs[ntype]
                deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
                adj_norm = adj / deg
                propagated = adj_norm @ raw_h[ntype]
                h_sim[ntype] = self.sim_proj[ntype](propagated)
            else:
                h_sim[ntype] = raw_h[ntype]

        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        for ntype in g.ntypes:
            if ntype not in h_orig or ntype not in h_sim:
                continue
            z1 = F.normalize(h_orig[ntype], dim=-1)
            z2 = F.normalize(h_sim[ntype], dim=-1)
            # InfoNCE-style loss
            sim_matrix = z1 @ z2.T / temperature
            labels = torch.arange(z1.size(0), device=z1.device)
            n = z1.size(0)
            if n > 2048:
                # Sample subset to avoid OOM
                idx = torch.randperm(n, device=z1.device)[:2048]
                sim_matrix = sim_matrix[idx][:, idx]
                labels = torch.arange(idx.size(0), device=z1.device)
            loss += F.cross_entropy(sim_matrix, labels)
            count += 1

        return loss / max(count, 1)

    def link_pred_loss(self, h, edge_splits, etype, split="train"):
        u, v = edge_splits[etype][split]
        src_ntype, dst_ntype = etype.split("2")
        u_feat = h[src_ntype][u]
        v_feat = h[dst_ntype][v]
        concat_feat = torch.cat([u_feat, v_feat], dim=1)
        pred = self.link_pred_heads[etype](concat_feat)

        pos_labels = torch.ones_like(pred)
        neg_u, neg_v = self._negative_sampling(u, v, h[src_ntype].shape[0], h[dst_ntype].shape[0])
        neg_u_feat = h[src_ntype][neg_u]
        neg_v_feat = h[dst_ntype][neg_v]
        neg_concat = torch.cat([neg_u_feat, neg_v_feat], dim=1)
        neg_pred = self.link_pred_heads[etype](neg_concat)
        neg_labels = torch.zeros_like(neg_pred)

        all_pred = torch.cat([pred, neg_pred], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)
        loss = F.binary_cross_entropy(all_pred, all_labels)
        return loss

    def _negative_sampling(self, u, v, num_src_nodes, num_dst_nodes, neg_ratio=1):
        num_pos = u.shape[0]
        num_neg = num_pos * neg_ratio
        neg_u = torch.randint(0, num_src_nodes, (num_neg,), device=u.device)
        neg_v = torch.randint(0, num_dst_nodes, (num_neg,), device=v.device)
        pos_set = set(zip(u.cpu().numpy(), v.cpu().numpy()))
        neg_set = set()
        for nu, nv in zip(neg_u.cpu().numpy(), neg_v.cpu().numpy()):
            if (nu, nv) not in pos_set:
                neg_set.add((nu, nv))
            if len(neg_set) == num_neg:
                break
        if len(neg_set) == 0:
            return neg_u[:1], neg_v[:1]
        neg_u, neg_v = zip(*neg_set)
        return torch.tensor(neg_u, device=u.device), torch.tensor(neg_v, device=v.device)


# ============================================================
#  Evaluation helpers (shared between baseline and denoised)
# ============================================================

def score_edges(model, h, etype, u, v):
    src_ntype, dst_ntype = etype.split("2")
    uv = torch.cat([h[src_ntype][u], h[dst_ntype][v]], dim=1)
    scores = model.link_pred_heads[etype](uv).detach().flatten().cpu().numpy()
    return scores


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
            pos_scores = score_edges(model, h, etype, pos_u, pos_v)
            neg_scores = score_edges(model, h, etype, neg_u.to(pos_u.device), neg_v.to(pos_v.device))
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
#  Graph loading and perturbation (reused from original)
# ============================================================

def load_base_graph(config):
    ds_cfg = config.dataset
    dataset = MacroNetDataset(**ds_cfg)
    try:
        dataset.load()
        g, _ = dataset[0]
    except Exception:
        dataset.process()
        g, _ = dataset[0]
    return g, dataset.pred_edge_types, float(ds_cfg.val_rate), float(ds_cfg.test_rate)


def random_remove_edges(g, drop_rate, rng):
    if drop_rate <= 0.0:
        return g
    g_new = g
    for canonical_etype in g_new.canonical_etypes:
        n_edges = g_new.num_edges(canonical_etype)
        if n_edges <= 1:
            continue
        n_drop = min(int(n_edges * drop_rate), n_edges - 1)
        if n_drop <= 0:
            continue
        drop_eids = rng.choice(n_edges, size=n_drop, replace=False)
        drop_eids = torch.as_tensor(drop_eids, dtype=g_new.idtype)
        g_new = dgl.remove_edges(g_new, drop_eids, etype=canonical_etype)
    return g_new


def random_remove_nodes(g, drop_rate, rng):
    if drop_rate <= 0.0:
        return g
    g_new = g
    for ntype in g_new.ntypes:
        n_nodes = g_new.num_nodes(ntype)
        if n_nodes <= 2:
            continue
        n_drop = min(int(n_nodes * drop_rate), n_nodes - 2)
        if n_drop <= 0:
            continue
        drop_nids = rng.choice(n_nodes, size=n_drop, replace=False)
        drop_nids = torch.as_tensor(drop_nids, dtype=g_new.idtype)
        g_new = dgl.remove_nodes(g_new, drop_nids, ntype=ntype)
    return g_new


def split_edges(g, pred_edge_types, val_rate, test_rate, rng, min_edges=30):
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


# ============================================================
#  Training one setting
# ============================================================

@dataclass
class EvalResult:
    model_type: str
    edge_drop_rate: float
    node_drop_rate: float
    repeat_id: int
    best_val_auc: float
    test_auc: float
    test_ap: float
    n_nodes: int
    n_edges: int


def train_one_setting(
    g_base, pred_edge_types, model_cfg, val_rate, test_rate,
    edge_drop_rate, node_drop_rate, repeat_seed, device,
    epochs, patience, lr, weight_decay, model_type="baseline",
    contrastive_weight=0.1, sim_k=15, alpha=0.7,
):
    rng = np.random.default_rng(repeat_seed)
    g = random_remove_edges(g_base, edge_drop_rate, rng)
    g = random_remove_nodes(g, node_drop_rate, rng)

    edge_splits = split_edges(g, pred_edge_types, val_rate, test_rate, rng)
    if len(edge_splits) == 0:
        return EvalResult(model_type, edge_drop_rate, node_drop_rate, repeat_seed,
                          float("nan"), float("nan"), float("nan"),
                          sum(g.num_nodes(nt) for nt in g.ntypes),
                          sum(g.num_edges(et) for et in g.etypes))

    g = g.to(device)
    edge_splits = {
        etype: {split: [uv[0].to(device), uv[1].to(device)] for split, uv in splits.items()}
        for etype, splits in edge_splits.items()
    }

    in_dims = {nt: g.nodes[nt].data["feat"].shape[1] for nt in g.ntypes}

    if model_type == "denoised":
        model = DenoisedMacroEncoder(
            in_dims=in_dims,
            hidden_dim=int(model_cfg.hidden_dim),
            num_layers=int(model_cfg.num_layers),
            num_heads=int(model_cfg.num_heads),
            dropout=float(model_cfg.dropout),
            sim_k=sim_k,
            alpha=alpha,
        ).to(device)
    else:
        model = MacroEncoder(
            in_dims=in_dims,
            **model_cfg,
        ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state, best_val_auc, bad_epochs = None, -1.0, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        h = model(g)

        losses = []
        for etype in edge_splits.keys():
            losses.append(model.link_pred_loss(h, edge_splits, etype, split="train"))
        if len(losses) == 0:
            break
        loss = torch.stack(losses).mean()

        # Add contrastive loss for denoised model
        if model_type == "denoised" and contrastive_weight > 0:
            cl = model.contrastive_loss(g)
            loss = loss + contrastive_weight * cl

        loss.backward()
        optimizer.step()

        val_auc, _ = evaluate_link_prediction(model, g, edge_splits, "val", rng)
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

    test_auc, test_ap = evaluate_link_prediction(model, g, edge_splits, "test", rng)
    return EvalResult(
        model_type=model_type,
        edge_drop_rate=edge_drop_rate,
        node_drop_rate=node_drop_rate,
        repeat_id=repeat_seed,
        best_val_auc=best_val_auc,
        test_auc=test_auc,
        test_ap=test_ap,
        n_nodes=sum(g.num_nodes(nt) for nt in g.ntypes),
        n_edges=sum(g.num_edges(et) for et in g.etypes),
    )


# ============================================================
#  Main
# ============================================================

def parse_rates(s):
    return [float(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser("Compare baseline vs denoised MacroEncoder quality")
    parser.add_argument("--config", type=str, default="configs/macro.yml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--edge-rates", type=str, default="0.0,0.05,0.1,0.2,0.3")
    parser.add_argument("--node-rates", type=str, default="0.0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--sim-k", type=int, default=15, help="kNN k for similarity graph")
    parser.add_argument("--alpha", type=float, default=0.7, help="fusion weight for original graph")
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--out-dir", type=str, default="output/denoise_quality")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    config = BaseConfig()
    config.load_from_file(args.config)
    seet_random_seed(args.seed)

    edge_rates = parse_rates(args.edge_rates)
    node_rates = parse_rates(args.node_rates)

    device = "cpu"
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.gpu}"

    g_base, pred_edge_types, val_rate, test_rate = load_base_graph(config)
    lr = float(args.lr if args.lr is not None else config.trainer.optimizer.lr)
    wd = float(args.weight_decay if args.weight_decay is not None else config.trainer.optimizer.weight_decay)

    all_results: List[EvalResult] = []

    model_types = ["baseline", "denoised"]
    total_runs = len(model_types) * len(edge_rates) * len(node_rates) * args.repeats
    run_idx = 0

    for mt in model_types:
        for er in edge_rates:
            for nr in node_rates:
                for rep in range(args.repeats):
                    run_idx += 1
                    repeat_seed = args.seed + rep
                    print(f"[{run_idx:03d}/{total_runs:03d}] {mt} | edge_drop={er:.2f}, node_drop={nr:.2f}, seed={repeat_seed}")
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
                        model_type=mt,
                        contrastive_weight=args.contrastive_weight,
                        sim_k=args.sim_k,
                        alpha=args.alpha,
                    )
                    all_results.append(res)
                    print(f"  -> val_auc={res.best_val_auc:.4f}, test_auc={res.test_auc:.4f}, test_ap={res.test_ap:.4f}")

    raw_df = pd.DataFrame([r.__dict__ for r in all_results])

    # Summary
    grp = raw_df.groupby(["model_type", "edge_drop_rate", "node_drop_rate"], as_index=False)
    summary = grp.agg(
        auc_mean=("test_auc", "mean"),
        auc_std=("test_auc", "std"),
        ap_mean=("test_ap", "mean"),
        ap_std=("test_ap", "std"),
        val_auc_mean=("best_val_auc", "mean"),
        runs=("repeat_id", "count"),
    )
    summary = summary.sort_values(["model_type", "edge_drop_rate", "node_drop_rate"])

    raw_fp = os.path.join(args.out_dir, "denoise_quality_raw.csv")
    summary_fp = os.path.join(args.out_dir, "denoise_quality_summary.csv")
    raw_df.to_csv(raw_fp, index=False)
    summary.to_csv(summary_fp, index=False)

    print("\n===== Comparison Summary =====")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved raw results to: {raw_fp}")
    print(f"Saved summary to: {summary_fp}")


if __name__ == "__main__":
    main()

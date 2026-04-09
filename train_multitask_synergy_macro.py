import argparse
import os

from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from my_config import BaseConfig
from models.datasets import SynergyDataset, MacroNetDataset
from models.models import SynergyBert, MacroEncoder
from models.utils import (
    seet_random_seed,
    get_logger,
    convert_to_bert_config,
    get_scheduler_by_name,
    kv_args,
    random_split_indices,
    count_model_params,
)


PROG_STR = "Train MultiTask (Synergy + Macro Link)"


def get_default_config(config_fp: str) -> BaseConfig:
    """Load config with safe defaults for multitask training.

    Defaults include task switches/weights and scheduler params, then values
    from the yaml file override these defaults.
    """
    config = BaseConfig()
    config.set_config_via_path("tasks.synergy.enabled", True)
    config.set_config_via_path("tasks.synergy.weight", 1.0)
    config.set_config_via_path("tasks.macro_link.enabled", True)
    config.set_config_via_path("tasks.macro_link.weight", 0.2)
    config.set_config_via_path("tasks.macro_link.every_n_steps", 1)

    config.set_config_via_path("trainer.scheduler.name", "constant")
    config.set_config_via_path("trainer.scheduler.params.num_training_steps", 1000)
    config.set_config_via_path("trainer.scheduler.params.num_warmup_steps", 100)

    config.load_from_file(config_fp)
    return config


def _to_device_edge_splits(edge_splits: Dict, device: str) -> Dict:
    """Move all edge index tensors in nested edge_splits dict to target device."""
    moved = {}
    for etype, splits in edge_splits.items():
        moved[etype] = {}
        for split, uv in splits.items():
            moved[etype][split] = [uv[0].to(device), uv[1].to(device)]
    return moved


def get_synergy_dataloaders(config: BaseConfig):
    """Build train/valid/test loaders for synergy regression.

    Uses explicit valid fold if provided; otherwise splits train folds into
    train/valid subsets by random key-based split.
    """
    dataset_cfg = config.dataset.synergy
    train_folds = dataset_cfg.train_folds
    valid_fold = dataset_cfg.get("valid_fold", None)
    test_fold = int(dataset_cfg.test_fold)

    train_set = SynergyDataset(dataset_cfg, use_folds=train_folds)
    if valid_fold is None:
        tr_indices, es_indices = random_split_indices(train_set, test_rate=0.1)
        valid_set = Subset(train_set, es_indices)
        train_set = Subset(train_set, tr_indices)
    else:
        valid_set = SynergyDataset(dataset_cfg, use_folds=[valid_fold])

    test_set = SynergyDataset(dataset_cfg, use_folds=[test_fold])

    train_loader = DataLoader(
        train_set,
        collate_fn=SynergyDataset.pad_batch,
        **dataset_cfg.train.loader,
    )
    valid_loader = DataLoader(
        valid_set,
        collate_fn=SynergyDataset.pad_batch,
        **dataset_cfg.valid.loader,
    )
    test_loader = DataLoader(
        test_set,
        collate_fn=SynergyDataset.pad_batch,
        **dataset_cfg.test.loader,
    )
    return train_loader, valid_loader, test_loader


def get_macro_graph(config: BaseConfig, device: str):
    """Load macro heterogeneous graph and edge splits on target device.

    Returns:
        graph: DGL heterograph
        edge_splits: dict[etype][split] -> [u_idx, v_idx]
        pred_edge_types: list of edge types used for link prediction
    """
    macro_ds_cfg = config.dataset.macro
    dataset = MacroNetDataset(**macro_ds_cfg)

    try:
        graph, edge_splits = dataset[0]
    except Exception:
        # Fallback for cases where cached graph is not initialized.
        dataset.process()
        graph, edge_splits = dataset[0]

    graph = graph.to(device)
    edge_splits = _to_device_edge_splits(edge_splits, device)
    return graph, edge_splits, dataset.pred_edge_types


def compute_macro_link_loss(
    macro_model: MacroEncoder,
    graph,
    edge_splits: Dict,
    etypes: List[str],
    split: str = "train",
):
    """Compute average link-prediction loss over all target edge types."""
    h = macro_model(graph)
    loss_list = []
    for etype in etypes:
        loss_list.append(macro_model.link_pred_loss(h, edge_splits, etype, split=split))
    return torch.stack(loss_list).mean()


def eval_synergy_only(model: SynergyBert, loader: DataLoader, device: str):
    """Evaluate synergy model on one loader using MSE and return predictions.

    Returns:
        total_loss: dataset-level mean MSE
        y_preds: flattened prediction list aligned with loader order
    """
    loss_fct = torch.nn.MSELoss()
    model.eval()
    total_loss = 0.0
    y_preds = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            labels = batch.pop("labels").view(-1, 1)
            pred = model(**batch)
            loss = loss_fct(pred, labels)
            total_loss += loss.item() * batch["drug_comb_ids"].size(0)
            y_preds.extend(pred.flatten().cpu().numpy())

    total_loss /= len(loader.dataset)
    return total_loss, y_preds


def run_fold(config: BaseConfig, logger):
    """Train one fold with joint optimization of main + auxiliary tasks.

    Main task: synergy regression (SynergyBert, MSE)
    Aux task: macro link prediction (MacroEncoder, BCE)
    Model selection uses validation loss of the main task.
    """
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    config.save_to_file(os.path.join(config.model_dir, "configs.yml"))

    if config.gpu < 0 or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = f"cuda:{config.gpu}"

    logger.info("Building datasets.")
    train_loader, valid_loader, test_loader = get_synergy_dataloaders(config)
    raw_test_samples = test_loader.dataset.raw_samples.copy()

    logger.info("Building models.")
    synergy_cfg = convert_to_bert_config(config.model.synergy)
    synergy_model = SynergyBert(synergy_cfg).to(device)

    macro_graph, edge_splits, pred_etypes = get_macro_graph(config, device)
    macro_model = MacroEncoder(
        in_dims={x: macro_graph.nodes[x].data["feat"].shape[1] for x in macro_graph.ntypes},
        **config.model.macro,
    ).to(device)

    n_total_s, n_trainable_s, n_freeze_s = count_model_params(synergy_model)
    n_total_m, n_trainable_m, n_freeze_m = count_model_params(macro_model)
    logger.info(
        f"Synergy Model Params: Total {n_total_s} | Trainable {n_trainable_s} | Freeze {n_freeze_s}"
    )
    logger.info(
        f"Macro Model Params: Total {n_total_m} | Trainable {n_trainable_m} | Freeze {n_freeze_m}"
    )

    trainer_cfg = config.trainer
    optimizer = AdamW(
        list(synergy_model.parameters()) + list(macro_model.parameters()),
        **trainer_cfg.optimizer,
    )
    scheduler = get_scheduler_by_name(trainer_cfg.scheduler.name, optimizer, **trainer_cfg.scheduler.params)

    loss_main_fct = torch.nn.MSELoss()
    w_main = float(config.tasks.synergy.weight)
    w_macro = float(config.tasks.macro_link.weight)
    macro_every = int(config.tasks.macro_link.every_n_steps)

    best_valid_loss = float("inf")
    best_epoch = -1
    best_test_loss = float("inf")
    best_preds = None
    angry = 0
    patience = trainer_cfg.patience
    global_step = 0

    for epc in range(1, trainer_cfg.num_epochs + 1):
        synergy_model.train()
        macro_model.train()
        ep_main_loss = 0.0
        ep_macro_loss = 0.0
        ep_total_loss = 0.0

        for batch in train_loader:
            global_step += 1
            optimizer.zero_grad()

            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            labels = batch.pop("labels").view(-1, 1)
            pred = synergy_model(**batch)
            loss_main = loss_main_fct(pred, labels)

            if bool(config.tasks.macro_link.enabled) and global_step % macro_every == 0:
                loss_macro = compute_macro_link_loss(
                    macro_model, macro_graph, edge_splits, pred_etypes, split="train"
                )
            else:
                loss_macro = torch.zeros(1, device=device).squeeze(0)

            loss_total = w_main * loss_main + w_macro * loss_macro
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            bs = batch["drug_comb_ids"].size(0)
            ep_main_loss += loss_main.item() * bs
            ep_macro_loss += float(loss_macro.item()) * bs
            ep_total_loss += loss_total.item() * bs

        ep_main_loss /= len(train_loader.dataset)
        ep_macro_loss /= len(train_loader.dataset)
        ep_total_loss /= len(train_loader.dataset)

        valid_loss, _ = eval_synergy_only(synergy_model, valid_loader, device)
        test_loss, y_preds = eval_synergy_only(synergy_model, test_loader, device)

        macro_val_loss = float("nan")
        if bool(config.tasks.macro_link.enabled):
            macro_model.eval()
            with torch.no_grad():
                macro_val_loss = compute_macro_link_loss(
                    macro_model, macro_graph, edge_splits, pred_etypes, split="val"
                ).item()

        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            best_test_loss = test_loss
            best_epoch = epc
            best_preds = y_preds
            angry = 0
        else:
            angry += 1

        logger.info(
            "Epoch %03d | Train Main %.4f | Train Macro %.4f | Train Total %.4f | "
            "Valid Main %.4f | Test Main %.4f | Macro Val %.4f | Best Epoch %03d"
            % (
                epc,
                ep_main_loss,
                ep_macro_loss,
                ep_total_loss,
                valid_loss,
                test_loss,
                macro_val_loss,
                best_epoch,
            )
        )

        if angry >= patience:
            logger.info("Valid main loss has not decreased for %d epochs, early stopped." % patience)
            break

    if best_preds is None:
        _, best_preds = eval_synergy_only(synergy_model, test_loader, device)

    raw_test_samples["prediction"] = best_preds
    out_fp_pred = os.path.join(config.model_dir, "predictions.csv")
    raw_test_samples.to_csv(out_fp_pred, sep="\t", index=False)
    logger.info("Best epoch %03d | Best valid main %.4f | Best test main %.4f" % (best_epoch, best_valid_loss, best_test_loss))
    logger.info("Saved predictions to: %s" % out_fp_pred)


def main(config: BaseConfig):
    """Entry for single-fold or all-fold training depending on config."""
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    logger = get_logger(PROG_STR, os.path.join(config.model_dir, "train.log"))

    if hasattr(config.dataset.synergy, "test_fold"):
        run_fold(config, logger)
    else:
        base_model_dir = config.model_dir
        for i in range(config.dataset.synergy.num_folds):
            config_tmp = deepcopy(config)
            config_tmp.model_dir = os.path.join(base_model_dir, str(i))
            config_tmp.dataset.synergy.test_fold = i
            run_fold(config_tmp, logger)


if __name__ == "__main__":
    """CLI entry: parse args, apply config overrides, set seed, then train."""
    parser = argparse.ArgumentParser(PROG_STR)
    parser.add_argument("config", type=str, help="config path")
    parser.add_argument("-s", "--sd", type=int, default=18)
    parser.add_argument("-u", "--update", type=kv_args, nargs="*", help="path.to.config=config_value")
    args = parser.parse_args()

    config = get_default_config(args.config)
    if args.update is not None:
        for k, v in args.update:
            config.set_config_via_path(k, v)

    seet_random_seed(args.sd)
    main(config)

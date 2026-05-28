"""Train recipe encoder + multitask heads.
Saves recipe embeddings and sustainability predictions for later use.
"""
import logging
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import copy
import torch.nn as nn
import numpy as np

from config import ModelConfig
from data import load_tables, IngredientTokenizer, RecipeDataset, TARGET_COLUMNS
from modules import RecipeTransformerEncoder, RecipeMoEEncoder, SustainabilityHead, multitask_loss


def train_recipe_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    tokenizer,
    cfg: ModelConfig,
    device: torch.device,
):
    def tensor_stats(tensor: torch.Tensor):
        t = tensor.detach()
        finite = torch.isfinite(t)
        if not finite.any():
            return {"min": None, "max": None, "mean": None}
        t = t[finite]
        return {
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "mean": float(t.mean().item()),
        }

    num_tasks = len(cfg.task_names)
    if cfg.task_weight_mode == "static":
        if len(cfg.task_weights) != num_tasks:
            raise ValueError("task_weights length must match task_names length")
        task_weights = torch.tensor(cfg.task_weights, device=device, dtype=torch.float32)
    else:
        task_weights = None
    task_log_vars = torch.zeros(num_tasks, device=device, dtype=torch.float32, requires_grad=True) if cfg.task_weight_mode == "learned" else None
    if cfg.num_experts > 1:
        model = RecipeMoEEncoder(
            vocab_size=len(tokenizer.token2id),
            embed_dim=cfg.recipe_embed_dim,
            num_heads=cfg.transformer_heads,
            num_layers=cfg.transformer_layers,
            dropout=cfg.transformer_dropout,
            quantity_dim=cfg.quantity_embed_dim,
            num_experts=cfg.num_experts,
            gate_hidden=cfg.moe_gate_hidden,
        ).to(device)
    else:
        model = RecipeTransformerEncoder(
            vocab_size=len(tokenizer.token2id),
            embed_dim=cfg.recipe_embed_dim,
            num_heads=cfg.transformer_heads,
            num_layers=cfg.transformer_layers,
            dropout=cfg.transformer_dropout,
            quantity_dim=cfg.quantity_embed_dim,
        ).to(device)
    head = SustainabilityHead(cfg.recipe_embed_dim, num_tasks=num_tasks).to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
        head = nn.DataParallel(head)
    optim_params = list(model.parameters()) + list(head.parameters())
    if task_log_vars is not None:
        optim_params.append(task_log_vars)
    optimizer = optim.AdamW(optim_params, lr=cfg.recipe_lr, weight_decay=cfg.recipe_weight_decay)

    model.train()
    head.train()
    if cfg.task_weight_mode == "static" and task_weights is not None:
        weight_sum = float(task_weights.sum().item())
        if weight_sum <= 0:
            weight_sum = 1.0
        normed = (task_weights / weight_sum).detach().cpu().tolist()
        weight_map = {name: float(w) for name, w in zip(cfg.task_names, normed)}
        logging.info("Static task_weights=%s", weight_map)
    for epoch in range(cfg.num_epochs_recipe):
        total_loss = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"Recipe epoch {epoch+1}/{cfg.num_epochs_recipe}")
        for batch in pbar:
            token_ids = batch["token_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            quantities = batch["quantities"].to(device)
            targets = batch["targets"].to(device)
            z = model(token_ids, quantities, attn_mask)
            s_hat = head(z)
            if not torch.isfinite(targets).all():
                logging.warning(
                    "Non-finite targets detected; skipping batch | targets=%s",
                    tensor_stats(targets),
                )
                continue
            if not torch.isfinite(s_hat).all():
                logging.warning(
                    "Non-finite predictions detected; skipping batch | z=%s s_hat=%s",
                    tensor_stats(z),
                    tensor_stats(s_hat),
                )
                continue
            loss = multitask_loss(
                s_hat,
                targets,
                loss_type=cfg.multitask_loss,
                task_weights=task_weights,
                task_weight_mode=cfg.task_weight_mode,
                task_log_vars=task_log_vars,
            )
            if torch.isnan(loss):
                logging.warning("NaN loss detected; skipping batch")
                continue
            optimizer.zero_grad()
            loss.backward()
            if cfg.recipe_max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(head.parameters()),
                    cfg.recipe_max_grad_norm,
                )
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({"loss": loss.item()})
        avg_loss = total_loss / max(steps, 1)
        val_loss = (
            evaluate_recipe(
                model,
                head,
                val_loader,
                device,
                cfg.multitask_loss,
                task_weights,
                cfg.task_weight_mode,
                task_log_vars,
            )
            if val_loader
            else 0.0
        )
        val_metrics = (
            evaluate_recipe_metrics(model, head, val_loader, device, cfg.multitask_loss, cfg.task_names)
            if val_loader
            else {}
        )
        if task_log_vars is not None:
            with torch.no_grad():
                weights = torch.exp(-task_log_vars).detach().cpu().tolist()
            weight_map = {name: float(w) for name, w in zip(cfg.task_names, weights)}
            logging.info("Epoch %d | task_weights=%s", epoch + 1, weight_map)
        logging.info("Epoch %d | train_loss=%.6f | val_loss=%.6f | val_metrics=%s", epoch + 1, avg_loss, val_loss, val_metrics)
        # Save checkpoint each epoch
        os.makedirs(cfg.checkpoints_dir, exist_ok=True)
        ckpt_path = os.path.join(cfg.checkpoints_dir, f"recipe_epoch_{epoch+1}.pt")
        torch.save(
            {
                "encoder": unwrap(model).state_dict(),
                "head": unwrap(head).state_dict(),
                "epoch": epoch + 1,
                "tok": tokenizer.token2id,
            },
            ckpt_path,
        )
    return unwrap(model), unwrap(head), task_log_vars


def precompute_recipe_embeddings(
    model: RecipeTransformerEncoder,
    recipes_loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in tqdm(recipes_loader, desc="Precompute recipe embeddings"):
            token_ids = batch["token_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            quantities = batch["quantities"].to(device)
            z = model(token_ids, quantities, attn_mask)
            embs.append(z.cpu())
    return torch.cat(embs, dim=0)


def evaluate_recipe(
    model: RecipeTransformerEncoder,
    head: SustainabilityHead,
    loader: DataLoader,
    device: torch.device,
    loss_type: str,
    task_weights: torch.Tensor,
    task_weight_mode: str,
    task_log_vars: torch.Tensor,
) -> float:
    if loader is None:
        return 0.0
    model.eval()
    head.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            quantities = batch["quantities"].to(device)
            targets = batch["targets"].to(device)
            z = model(token_ids, quantities, attn_mask)
            s_hat = head(z)
            loss = multitask_loss(
                s_hat,
                targets,
                loss_type=loss_type,
                task_weights=task_weights,
                task_weight_mode=task_weight_mode,
                task_log_vars=task_log_vars,
            )
            losses.append(loss.item())
    model.train()
    head.train()
    return float(sum(losses) / max(len(losses), 1))


def evaluate_recipe_metrics(
    model: RecipeTransformerEncoder,
    head: SustainabilityHead,
    loader: DataLoader,
    device: torch.device,
    loss_type: str,
    task_names,
):
    if loader is None:
        return {}
    model.eval()
    head.eval()
    num_tasks = len(task_names)
    sums_mae = torch.zeros(num_tasks)
    sums_mse = torch.zeros(num_tasks)
    all_targets = []
    all_preds = []
    count = 0
    with torch.no_grad():
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            quantities = batch["quantities"].to(device)
            targets = batch["targets"].to(device)
            z = model(token_ids, quantities, attn_mask)
            s_hat = head(z)
            if not torch.isfinite(targets).all() or not torch.isfinite(s_hat).all():
                continue
            all_targets.append(targets.cpu())
            all_preds.append(s_hat.cpu())
            sums_mae += torch.mean(torch.abs(s_hat - targets), dim=0).cpu()
            sums_mse += torch.mean((s_hat - targets) ** 2, dim=0).cpu()
            count += 1
    model.train()
    head.train()
    mae_vals = (sums_mae / max(count, 1)).tolist()
    mse_vals = (sums_mse / max(count, 1)).tolist()
    rmse_vals = [m ** 0.5 for m in mse_vals]
    if all_targets:
        targets_all = torch.cat(all_targets, dim=0)
        preds_all = torch.cat(all_preds, dim=0)
        r2_vals = []
        for i in range(targets_all.shape[1]):
            t = targets_all[:, i]
            p = preds_all[:, i]
            ss_tot = torch.sum((t - torch.mean(t)) ** 2)
            ss_res = torch.sum((t - p) ** 2)
            r2 = 0.0 if ss_tot.item() == 0 else float(1 - (ss_res / ss_tot).item())
            r2_vals.append(r2)
    else:
        r2_vals = [0.0] * num_tasks
    metrics = {}
    for i, name in enumerate(task_names):
        metrics[name] = {"mae": mae_vals[i], "rmse": rmse_vals[i], "r2": r2_vals[i]}
    return metrics

def main():
    cfg = ModelConfig()
    os.makedirs(cfg.logs_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg.logs_dir, "train_recipes.log")),
            logging.StreamHandler(),
        ],
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    if cfg.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        logging.info("Autograd anomaly detection enabled")
    logging.info(
        "Recipe stage hyperparams: batch_size=%d lr=%g weight_decay=%g max_grad_norm=%g epochs=%d",
        cfg.recipe_batch_size,
        cfg.recipe_lr,
        cfg.recipe_weight_decay,
        cfg.recipe_max_grad_norm,
        cfg.num_epochs_recipe,
    )
    recipes, ingredients, interactions = load_tables(cfg.recipes_path, cfg.ingredients_path, cfg.interactions_path)
    # Data sanity checks
    for col in TARGET_COLUMNS:
        series = pd.to_numeric(recipes[col], errors="coerce")
        nan_count = series.isna().sum()
        inf_count = np.isinf(series).sum()
        logging.info("Column %s: nan=%d inf=%d", col, int(nan_count), int(inf_count))
    qty_series = pd.to_numeric(ingredients["weight_total_g"], errors="coerce")
    logging.info("weight_total_g: nan=%d inf=%d max=%.4f", int(qty_series.isna().sum()), int(np.isinf(qty_series).sum()), float(qty_series.max()))

    # Split recipes 80/10/10 by recipe_id and subset ingredients accordingly, but reuse if split exists
    split_path = os.path.join(cfg.logs_dir, "recipe_splits.csv")
    os.makedirs(cfg.logs_dir, exist_ok=True)
    if os.path.exists(split_path):
        splits_df = pd.read_csv(split_path)
        train_ids = splits_df[splits_df["split"] == "train"]["recipe_id"]
        val_ids = splits_df[splits_df["split"] == "val"]["recipe_id"]
        test_ids = splits_df[splits_df["split"] == "test"]["recipe_id"]
        train_recipes = recipes[recipes["recipe_id"].isin(train_ids)]
        val_recipes = recipes[recipes["recipe_id"].isin(val_ids)]
        test_recipes = recipes[recipes["recipe_id"].isin(test_ids)]
    else:
        recipes_shuffled = recipes.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n_total = len(recipes_shuffled)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        train_recipes = recipes_shuffled.iloc[:n_train]
        val_recipes = recipes_shuffled.iloc[n_train : n_train + n_val]
        test_recipes = recipes_shuffled.iloc[n_train + n_val :]
        splits = []
        for rid in train_recipes["recipe_id"]:
            splits.append({"recipe_id": rid, "split": "train"})
        for rid in val_recipes["recipe_id"]:
            splits.append({"recipe_id": rid, "split": "val"})
        for rid in test_recipes["recipe_id"]:
            splits.append({"recipe_id": rid, "split": "test"})
        pd.DataFrame(splits).to_csv(split_path, index=False)

    def subset_ing(rec_ids):
        return ingredients[ingredients["recipe_id"].isin(rec_ids)]

    tokenizer = IngredientTokenizer(min_freq=cfg.ingredient_vocab_min_freq)
    tokenizer.build_vocab(subset_ing(train_recipes["recipe_id"]))

    train_ds = RecipeDataset(
        train_recipes,
        subset_ing(train_recipes["recipe_id"]),
        tokenizer,
        cfg.max_ingredients,
        cfg.animal_alpha,
        task_names=cfg.task_names,
    )
    # Use train-set normalization for all splits to keep metrics comparable.
    normalizer = train_ds.normalizer
    val_ds = RecipeDataset(
        val_recipes,
        subset_ing(val_recipes["recipe_id"]),
        tokenizer,
        cfg.max_ingredients,
        cfg.animal_alpha,
        task_names=cfg.task_names,
        normalizer=normalizer,
    )
    test_ds = RecipeDataset(
        test_recipes,
        subset_ing(test_recipes["recipe_id"]),
        tokenizer,
        cfg.max_ingredients,
        cfg.animal_alpha,
        task_names=cfg.task_names,
        normalizer=normalizer,
    )
    full_ds = RecipeDataset(
        recipes,
        ingredients,
        tokenizer,
        cfg.max_ingredients,
        cfg.animal_alpha,
        task_names=cfg.task_names,
        normalizer=normalizer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_ds, batch_size=cfg.recipe_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.recipe_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.recipe_batch_size, shuffle=False, num_workers=0)

    encoder, head, task_log_vars = train_recipe_model(train_loader, val_loader, test_loader, tokenizer, cfg, device)

    loader_full = DataLoader(full_ds, batch_size=cfg.recipe_batch_size, shuffle=False)
    z_table = precompute_recipe_embeddings(encoder, loader_full, device)

    head.eval()
    s_hat_table = []
    with torch.no_grad():
        for batch in loader_full:
            z = encoder(
                batch["token_ids"].to(device),
                batch["quantities"].to(device),
                batch["attn_mask"].to(device),
            )
            s_hat_table.append(head(z).cpu())
    s_hat_table = torch.cat(s_hat_table, dim=0)

    # Evaluate test metrics per task
    test_metrics = evaluate_recipe_metrics(encoder, head, test_loader, device, cfg.multitask_loss, cfg.task_names)

    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "z": z_table,
            "s_hat": s_hat_table,
            "tok": tokenizer.token2id,
            "task_names": cfg.task_names,
            "task_weights": cfg.task_weights,
            "task_weight_mode": cfg.task_weight_mode,
            "task_log_vars": task_log_vars.detach().cpu().tolist() if task_log_vars is not None else None,
        },
        os.path.join(cfg.checkpoints_dir, "recipe_stage.pt"),
    )
    if cfg.task_weight_mode == "static":
        task_weights = torch.tensor(cfg.task_weights, device=device, dtype=torch.float32)
    else:
        task_weights = None
    test_loss = evaluate_recipe(
        encoder, head, test_loader, device, cfg.multitask_loss, task_weights, cfg.task_weight_mode, task_log_vars
    )
    logging.info(
        "Recipe training completed. Saved to %s",
        os.path.join(cfg.checkpoints_dir, "recipe_stage.pt"),
    )
    logging.info("Test loss: %.4f", test_loss)
    logging.info("Test metrics per task (MAE/RMSE): %s", test_metrics)


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


if __name__ == "__main__":
    main()

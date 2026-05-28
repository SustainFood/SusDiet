import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ModelConfig
from data import load_tables, IngredientTokenizer, RecipeDataset, signed_log1p
from modules import RecipeTransformerEncoder, RecipeMoEEncoder, SustainabilityHead
from recipe_encoder import precompute_recipe_embeddings


def compute_welfare_stats(recipes_df, normalizer, animal_alpha: float):
    lives_stats = normalizer.get("total_lives_per_serving")
    suffer_stats = normalizer.get("total_suffering_per_serving")
    if lives_stats is None or suffer_stats is None:
        return 0.0, 1.0

    lives_vals = pd.to_numeric(recipes_df["total_lives_per_serving"], errors="coerce").to_numpy()
    suffer_vals = pd.to_numeric(recipes_df["total_suffering_per_serving"], errors="coerce").to_numpy()

    def norm_vals(values, stats):
        mean, std, lower, upper = stats
        if std == 0 or np.isnan(std):
            return np.zeros_like(values, dtype=np.float32)
        vals = signed_log1p(values)
        vals = np.clip(vals, lower, upper)
        return (vals - mean) / std

    lives_norm = norm_vals(lives_vals, lives_stats)
    suffer_norm = norm_vals(suffer_vals, suffer_stats)
    welfare = animal_alpha * lives_norm + (1.0 - animal_alpha) * suffer_norm
    welfare = np.nan_to_num(welfare, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(np.mean(welfare))
    std = float(np.std(welfare))
    if std == 0 or np.isnan(std):
        std = 1.0
    return mean, std


def build_s_hat_stats(recipes, ingredients, tokenizer, cfg: ModelConfig):
    split_path = os.path.join(cfg.logs_dir, "recipe_splits.csv")
    if os.path.exists(split_path):
        splits_df = pd.read_csv(split_path)
        train_ids = splits_df[splits_df["split"] == "train"]["recipe_id"]
        train_recipes = recipes[recipes["recipe_id"].isin(train_ids)]
        train_ingredients = ingredients[ingredients["recipe_id"].isin(train_ids)]
        train_ds = RecipeDataset(
            train_recipes,
            train_ingredients,
            tokenizer,
            cfg.max_ingredients,
            cfg.animal_alpha,
            task_names=cfg.task_names,
        )
        normalizer = train_ds.normalizer
        stats_recipes = train_recipes
    else:
        full_ds = RecipeDataset(
            recipes,
            ingredients,
            tokenizer,
            cfg.max_ingredients,
            cfg.animal_alpha,
            task_names=cfg.task_names,
        )
        normalizer = full_ds.normalizer
        stats_recipes = recipes

    means = []
    stds = []
    for task in cfg.task_names:
        if task == "welfare":
            mean, std = compute_welfare_stats(stats_recipes, normalizer, cfg.animal_alpha)
            means.append(mean)
            stds.append(std)
            continue
        if task in normalizer:
            mean, std, _, _ = normalizer[task]
            means.append(float(mean))
            stds.append(float(std))
        else:
            means.append(0.0)
            stds.append(1.0)
    return means, stds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch_ckpt",
        type=str,
        default="checkpoints/recipe_epoch_131.pt",
        help="Path to recipe_epoch_x.pt",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for recipe_stage.pt (defaults to checkpoints/recipe_stage.pt)",
    )
    args = parser.parse_args()

    cfg = ModelConfig()
    out_path = args.out or os.path.join(cfg.checkpoints_dir, "recipe_stage131.pt")
    if not os.path.exists(args.epoch_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.epoch_ckpt}")

    recipes, ingredients, _ = load_tables(cfg.recipes_path, cfg.ingredients_path, cfg.interactions_path)
    ckpt = torch.load(args.epoch_ckpt, map_location="cpu")
    tokenizer = IngredientTokenizer(min_freq=cfg.ingredient_vocab_min_freq)
    if "tok" in ckpt:
        tokenizer.token2id = ckpt["tok"]
    else:
        split_path = os.path.join(cfg.logs_dir, "recipe_splits.csv")
        if os.path.exists(split_path):
            splits_df = pd.read_csv(split_path)
            train_ids = splits_df[splits_df["split"] == "train"]["recipe_id"]
            train_ingredients = ingredients[ingredients["recipe_id"].isin(train_ids)]
            tokenizer.build_vocab(train_ingredients)
        else:
            tokenizer.build_vocab(ingredients)

    full_ds = RecipeDataset(
        recipes,
        ingredients,
        tokenizer,
        cfg.max_ingredients,
        cfg.animal_alpha,
        task_names=cfg.task_names,
    )
    loader = DataLoader(full_ds, batch_size=cfg.recipe_batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.num_experts > 1:
        encoder = RecipeMoEEncoder(
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
        encoder = RecipeTransformerEncoder(
            vocab_size=len(tokenizer.token2id),
            embed_dim=cfg.recipe_embed_dim,
            num_heads=cfg.transformer_heads,
            num_layers=cfg.transformer_layers,
            dropout=cfg.transformer_dropout,
            quantity_dim=cfg.quantity_embed_dim,
        ).to(device)

    head = SustainabilityHead(cfg.recipe_embed_dim, num_tasks=len(cfg.task_names)).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])

    z = precompute_recipe_embeddings(encoder, loader, device)

    head.eval()
    s_hat_list = []
    with torch.no_grad():
        for batch in loader:
            z_batch = encoder(
                batch["token_ids"].to(device),
                batch["quantities"].to(device),
                batch["attn_mask"].to(device),
            )
            s_hat_list.append(head(z_batch).cpu())
    s_hat = torch.cat(s_hat_list, dim=0)
    s_hat_mean, s_hat_std = build_s_hat_stats(recipes, ingredients, tokenizer, cfg)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "z": z,
            "s_hat": s_hat,
            "s_hat_mean": s_hat_mean,
            "s_hat_std": s_hat_std,
            "tok": tokenizer.token2id,
            "task_names": cfg.task_names,
            "task_weights": cfg.task_weights,
            "task_weight_mode": cfg.task_weight_mode,
            "task_log_vars": None,
        },
        out_path,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

"""Evaluate recipe encoder + multitask heads on saved val/test splits."""
import argparse
import logging
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from config import ModelConfig
from data import IngredientTokenizer, RecipeDataset, load_tables
from modules import RecipeTransformerEncoder, RecipeMoEEncoder, SustainabilityHead
from recipe_encoder import evaluate_recipe_metrics


def main(split: str, ckpt_path: str):
    assert split in {"val", "test"}, "split must be 'val' or 'test'"
    cfg = ModelConfig()
    os.makedirs(cfg.logs_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg.logs_dir, f"eval_recipes_{split}.log")),
            logging.StreamHandler(),
        ],
    )
    recipes, ingredients, _ = load_tables(cfg.recipes_path, cfg.ingredients_path, cfg.interactions_path)
    splits = pd.read_csv(os.path.join(cfg.logs_dir, "recipe_splits.csv"))
    train_ids = splits[splits["split"] == "train"]["recipe_id"].tolist()
    target_ids = splits[splits["split"] == split]["recipe_id"].tolist()
    train_recipes = recipes[recipes["recipe_id"].isin(train_ids)]
    train_ingredients = ingredients[ingredients["recipe_id"].isin(train_ids)]
    subset_recipes = recipes[recipes["recipe_id"].isin(target_ids)]
    subset_ingredients = ingredients[ingredients["recipe_id"].isin(target_ids)]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    tokenizer = IngredientTokenizer(min_freq=cfg.ingredient_vocab_min_freq)
    if "tok" in ckpt:
        tokenizer.token2id = ckpt["tok"]
    else:
        tokenizer.build_vocab(train_ingredients)

    task_names = ckpt.get("task_names", cfg.task_names)
    if task_names != cfg.task_names:
        logging.info("Using task_names from checkpoint: %s", task_names)
    train_ds = RecipeDataset(
        train_recipes,
        train_ingredients,
        tokenizer,
        cfg.max_ingredients,
        cfg.animal_alpha,
        task_names=task_names,
    )

    ds = RecipeDataset(
        subset_recipes,
        subset_ingredients,
        tokenizer,
        cfg.max_ingredients,
        cfg.animal_alpha,
        task_names=task_names,
        normalizer=train_ds.normalizer,
    )
    loader = DataLoader(ds, batch_size=cfg.recipe_batch_size, shuffle=False)
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
    head = SustainabilityHead(cfg.recipe_embed_dim, num_tasks=len(task_names)).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])

    metrics = evaluate_recipe_metrics(encoder, head, loader, device, cfg.multitask_loss, task_names)
    logging.info("Split: %s | Metrics: %s", split, metrics)


if __name__ == "__main__":
    cfg_default = ModelConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(cfg_default.checkpoints_dir, "recipe_stage.pt"),
    )
    args = parser.parse_args()
    main(args.split, args.ckpt)

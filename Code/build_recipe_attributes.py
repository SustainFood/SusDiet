import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def load_token_embedding(ckpt_path: str) -> Tuple[Dict[str, int], np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tok = ckpt.get("tok")
    if tok is None:
        raise ValueError("token mapping 'tok' not found in recipe_stage checkpoint.")
    state = ckpt.get("encoder")
    if state is None:
        raise ValueError("encoder state not found in recipe_stage checkpoint.")
    token_key = "token_embed.weight"
    if token_key not in state:
        raise ValueError(f"token embedding weights not found: {token_key}")
    emb = state[token_key].detach().cpu().numpy()
    return tok, emb


def build_ingredient_index(
    ingredients: pd.DataFrame, tok: Dict[str, int]
) -> Dict[int, List[Tuple[int, float]]]:
    by_recipe: Dict[int, List[Tuple[int, float]]] = {}
    unk_id = tok.get("<unk>", 1)
    for row in ingredients.itertuples(index=False):
        try:
            rid = int(row.recipe_id)
        except Exception:
            continue
        name = str(row.ingredient_std).lower()
        idx = tok.get(name, unk_id)
        weight = getattr(row, "weight_total_g", None)
        if weight is None or weight != weight:
            weight = 1.0
        else:
            weight = float(weight)
            if weight <= 0:
                weight = 1.0
        by_recipe.setdefault(rid, []).append((idx, weight))
    return by_recipe


def compute_attr_vectors(
    recipes: pd.DataFrame,
    by_recipe: Dict[int, List[Tuple[int, float]]],
    token_emb: np.ndarray,
) -> np.ndarray:
    dim = token_emb.shape[1]
    attr = np.zeros((len(recipes), dim), dtype=np.float32)
    for i, rid in enumerate(recipes["recipe_id"].tolist()):
        items = by_recipe.get(int(rid), [])
        if not items:
            continue
        ids, weights = zip(*items)
        ids = np.array(ids, dtype=np.int64)
        w = np.array(weights, dtype=np.float32)
        w = w / (w.sum() + 1e-8)
        vecs = token_emb[ids]
        attr[i] = (vecs * w[:, None]).sum(axis=0)
    return attr


def main() -> None:
    parser = argparse.ArgumentParser(description="Build recipe attribute vectors from ingredient_std.")
    parser.add_argument("--recipes", type=str, default="Data/all_recipes.xlsx")
    parser.add_argument("--ingredients", type=str, default="Data/all_ingredients.csv")
    parser.add_argument("--ckpt", type=str, default="checkpoints/recipe_stage131.pt")
    parser.add_argument("--output", type=str, default="logs/recipe_attr.npy")
    args = parser.parse_args()

    recipes = pd.read_excel(args.recipes)
    if "recipe_id" not in recipes.columns:
        raise ValueError("Missing recipe_id column in recipes.")
    ingredients = pd.read_csv(args.ingredients)
    tok, token_emb = load_token_embedding(args.ckpt)
    by_recipe = build_ingredient_index(ingredients, tok)
    attr = compute_attr_vectors(recipes, by_recipe, token_emb)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, attr)
    print(f"Saved attribute vectors: {args.output} shape={attr.shape}")


if __name__ == "__main__":
    main()

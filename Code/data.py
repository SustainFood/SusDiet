"""
Data loading and preprocessing helpers.
- Builds ingredient vocabulary.
- Creates datasets for recipe multitask training and user-recipe interactions.
"""
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional


TARGET_COLUMNS = [
    "price_total",
    "NutriScore_Scaled",
    "Tot_env_100g",
    "total_lives_per_serving",
    "total_suffering_per_serving",
]

ALL_TASKS = ["price_total", "NutriScore_Scaled", "Tot_env_100g", "welfare"]
LOG1P_TARGETS = set(TARGET_COLUMNS)
CLIP_QUANTILES = (0.0, 0.99)


def signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))


class IngredientTokenizer:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.token2id: Dict[str, int] = {"<pad>": 0, "<unk>": 1}

    def build_vocab(self, ingredients: pd.DataFrame) -> None:
        counter = Counter(ingredients["ingredient_std"].astype(str).str.lower())
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token2id:
                self.token2id[token] = len(self.token2id)

    def encode(self, tokens: List[str], max_len: int) -> Tuple[List[int], List[int]]:
        ids = [self.token2id.get(t.lower(), 1) for t in tokens][:max_len]
        pad_len = max_len - len(ids)
        ids += [0] * pad_len
        attn = [1] * (len(ids) - pad_len) + [0] * pad_len
        return ids, attn


class RecipeDataset(Dataset):
    def __init__(
        self,
        recipes: pd.DataFrame,
        ingredients: pd.DataFrame,
        tokenizer: IngredientTokenizer,
        max_len: int,
        animal_alpha: float,
        task_names: Optional[List[str]] = None,
        normalizer: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    ):
        self.recipes = recipes
        self.ingredients = ingredients
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.animal_alpha = animal_alpha
        self.task_names = task_names or ALL_TASKS
        invalid = [t for t in self.task_names if t not in ALL_TASKS]
        if invalid:
            raise ValueError(f"Unsupported task names: {invalid}")
        self.normalizer = normalizer or self._compute_normalizer()
        self.recipe_to_ing = self._group_ingredients()

    def _group_ingredients(self) -> Dict[int, pd.DataFrame]:
        grouped = defaultdict(list)
        for _, row in self.ingredients.iterrows():
            grouped[int(row["recipe_id"])].append(row)
        return {k: pd.DataFrame(v) for k, v in grouped.items()}

    def _compute_normalizer(self) -> Dict[str, Tuple[float, float, float, float]]:
        stats = {}
        for col in TARGET_COLUMNS:
            vals = pd.to_numeric(self.recipes[col], errors="coerce")
            vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
            if col in LOG1P_TARGETS and len(vals) > 0:
                vals = pd.Series(signed_log1p(vals.to_numpy()))
            if len(vals) == 0:
                mean, std = 0.0, 1.0
                lower, upper = 0.0, 0.0
            else:
                lower = float(vals.quantile(CLIP_QUANTILES[0]))
                upper = float(vals.quantile(CLIP_QUANTILES[1]))
                vals = vals.clip(lower=lower, upper=upper)
                mean = float(vals.mean())
                std = float(vals.std())
                if std == 0 or np.isnan(std):
                    std = 1.0
            stats[col] = (mean, std, lower, upper)
        return stats

    def __len__(self) -> int:
        return len(self.recipes)

    def __getitem__(self, idx: int):
        row = self.recipes.iloc[idx]
        recipe_id = int(row["recipe_id"])
        ing_df = self.recipe_to_ing.get(recipe_id, pd.DataFrame(columns=["ingredient_std", "quantity"]))
        tokens = ing_df["ingredient_std"].astype(str).tolist()
        quantities = ing_df["weight_total_g"].fillna(0).astype(float).tolist()
        token_ids, attn = self.tokenizer.encode(tokens, self.max_len)
        quantities = quantities[: self.max_len]
        quantities += [0.0] * (self.max_len - len(quantities))

        targets = {col: float(row[col]) for col in TARGET_COLUMNS}
        welfare = self.animal_alpha * self._norm("total_lives_per_serving", targets) + (
            1 - self.animal_alpha
        ) * self._norm("total_suffering_per_serving", targets)

        values = []
        for task in self.task_names:
            if task == "welfare":
                values.append(welfare)
            else:
                values.append(self._norm(task, targets))
        y = torch.tensor(values, dtype=torch.float32)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        quantities = torch.tensor(quantities, dtype=torch.float32)
        quantities = torch.nan_to_num(quantities, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "recipe_id": recipe_id,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "attn_mask": torch.tensor(attn, dtype=torch.float32),
            "quantities": quantities,
            "targets": y,
        }

    def _norm(self, key: str, targets: Dict[str, float]) -> float:
        mean, std, lower, upper = self.normalizer[key]
        if std == 0:
            return 0.0
        value = targets[key]
        if key in LOG1P_TARGETS:
            value = float(signed_log1p(value))
        value = float(min(max(value, lower), upper))
        return (value - mean) / std


class InteractionDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        recipe_id_to_index: Dict[int, int],
        user_id_to_index: Optional[Dict[int, int]] = None,
        culture_scores: Optional[pd.DataFrame] = None,
        rating_mean: Optional[float] = None,
        rating_std: Optional[float] = None,
    ):
        self.interactions = interactions
        self.recipe_id_to_index = recipe_id_to_index
        self.user_id_to_index = user_id_to_index or self._build_user_mapping()
        self.culture = self._merge_culture(culture_scores)
        self.rating_mean = rating_mean
        self.rating_std = rating_std

    def _build_user_mapping(self) -> Dict[int, int]:
        unique_users = sorted(self.interactions["user_id"].unique())
        return {int(u): i for i, u in enumerate(unique_users)}

    def _merge_culture(self, culture_scores: Optional[pd.DataFrame]) -> Dict[Tuple[int, int], float]:
        mapping = {}
        if culture_scores is None:
            return mapping
        for _, row in culture_scores.iterrows():
            key = (int(row["user_id"]), int(row["recipe_id"]))
            mapping[key] = float(row["culture_score"])
        return mapping

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int):
        row = self.interactions.iloc[idx]
        user_id = int(row["user_id"])
        recipe_id = int(row["recipe_id"])
        rating_raw = float(row["rating"])
        if self.rating_mean is not None and self.rating_std not in (None, 0):
            rating = (rating_raw - self.rating_mean) / self.rating_std
        else:
            rating = rating_raw
        culture_score = float(self.culture.get((user_id, recipe_id), 0.0))
        return {
            "user_idx": self.user_id_to_index[user_id],
            "recipe_idx": self.recipe_id_to_index[recipe_id],
            "rating": torch.tensor(rating, dtype=torch.float32),
            "rating_raw": torch.tensor(rating_raw, dtype=torch.float32),
            "culture_score": torch.tensor(culture_score, dtype=torch.float32),
        }


def load_tables(
    recipes_path: str,
    ingredients_path: str,
    interactions_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recipes = pd.read_excel(recipes_path)
    ingredients = pd.read_csv(ingredients_path)
    interactions = pd.read_csv(interactions_path)
    interactions = interactions.rename(columns={"score": "rating", "ratings": "rating"})
    interactions["rating"] = interactions["rating"].fillna(0)
    return recipes, ingredients, interactions


def load_recipes_interactions(
    recipes_path: str,
    interactions_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    recipes = pd.read_excel(recipes_path)
    interactions = pd.read_csv(interactions_path)
    interactions = interactions.rename(columns={"score": "rating", "ratings": "rating"})
    interactions["rating"] = interactions["rating"].fillna(0)
    return recipes, interactions

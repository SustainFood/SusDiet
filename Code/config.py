"""
Configuration defaults for the sustainable recipe recommender.
Modify these values or override via argparse in scripts.
"""

from dataclasses import dataclass, field
import os


@dataclass
class ModelConfig:
    recipe_embed_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_dropout: float = 0.1
    max_ingredients: int = 64
    ingredient_vocab_min_freq: int = 2
    quantity_embed_dim: int = 16
    num_experts: int = 4
    moe_gate_hidden: int = 64

    # Multitask targets
    multitask_loss: str = "rmse"  # or "mae"
    animal_alpha: float = 0.5
    task_names: list = field(
        default_factory=lambda: ["price_total", "NutriScore_Scaled", "Tot_env_100g", "welfare"]
    )
    task_weights: list = field(default_factory=lambda: [1.0, 3.0, 3.0, 1.0])
    task_weight_mode: str = "learned"  # "static" or "learned"

    # Training
    recipe_batch_size: int = 64
    recipe_lr: float = 1e-4
    recipe_weight_decay: float = 1e-4
    recipe_max_grad_norm: float = 0.5
    num_epochs_recipe: int = 150
    detect_anomaly: bool = False
    gpu_ids: str = "5"  # comma-separated GPU ids, e.g., "0,1"

    # Paths
    data_dir: str = "Data"
    logs_dir: str = "logs"
    checkpoints_dir: str = "checkpoints"

    recipes_path: str = os.path.join(data_dir, "all_recipes.xlsx")
    ingredients_path: str = os.path.join(data_dir, "all_ingredients.csv")
    interactions_path: str = os.path.join(data_dir, "all_interaction.csv")

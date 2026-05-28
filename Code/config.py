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

    # User modeling
    user_embed_dim: int = 128
    attn_hidden_dim: int = 64
    user_num_interests: int = 4
    user_rating_embed_dim: int = 8
    user_set_hidden_dim: int = 128
    user_cf_dim: int = 128
    user_batch_by_user: bool = True
    user_batch_per_user: int = 2
    culture_lambda: float = 0.0 # lambda cul
    sustain_lambda: float = 0.0 # lambda sus
    sustain_loss_alpha: float = 0.0 # alpha sus
    sustain_softmax_temp: float = 1.0
    tolerance_l2_alpha: float = 0.0000 # alpha tol
    rating_standardize: bool = True
    min_rank_interactions: int = 2
    bpr_alpha: float = 1.0
    bpr_pos_threshold: float = 3.0
    bpr_negatives_per_pos: int = 1

    # Training
    recipe_batch_size: int = 64
    interaction_batch_size: int = 64
    recipe_lr: float = 1e-4
    interaction_lr: float = 5e-4
    recipe_weight_decay: float = 1e-4
    interaction_weight_decay: float = 1e-4
    recipe_max_grad_norm: float = 0.5
    interaction_max_grad_norm: float = 1.0
    num_epochs_recipe: int = 150
    num_epochs_interaction: int = 150
    pairwise_rank_alpha: float = 1 # alpha rank
    min_interactions_per_user: int = 1
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    rank_loss_history_pairs: int = 3
    rank_pairs_per_user: int = 2
    rank_pos_threshold: float = 3.0
    rank_neg_threshold: float = 2.9
    rank_eval_candidates: int = 5000
    rank_eval_users: int = 0
    rank_eval_exclude_history: bool = True
    rank_eval_use_culture: bool = True
    rank_eval_interval: int = 1
    detect_anomaly: bool = False
    gpu_ids: str = "5"  # comma-separated GPU ids, e.g., "0,1"

    # Paths
    data_dir: str = "Data"
    logs_dir: str = "logs"
    checkpoints_dir: str = "checkpoints"

    recipes_path: str = os.path.join(data_dir, "all_recipes.xlsx")
    ingredients_path: str = os.path.join(data_dir, "all_ingredients.csv")
    interactions_path: str = os.path.join(data_dir, "all_interaction.csv")
    culture_scores_path: str = os.path.join(
        data_dir, "culture_scores.csv"
    )  # optional csv with columns user_id, recipe_id, culture_score

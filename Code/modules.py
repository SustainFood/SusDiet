"""
Model modules:
- RecipeTransformerEncoder: ingredient embedding + transformer to recipe vector.
- RecipeMoEEncoder: mixture-of-experts encoder for recipe embeddings.
- SustainabilityHead: four-task regression heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class RecipeTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        quantity_dim: int,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.quantity_proj = nn.Linear(1, quantity_dim)
        self.input_proj = nn.Linear(embed_dim + quantity_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: torch.Tensor, quantities: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [B, L], quantities: [B, L], attn_mask: [B, L]
        returns recipe embedding [B, D]
        """
        tok = self.token_embed(token_ids)
        qty = self.quantity_proj(quantities.unsqueeze(-1))
        x = torch.cat([tok, qty], dim=-1)
        x = self.input_proj(x)
        key_padding_mask = attn_mask == 0
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        masked = h * attn_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / (attn_mask.sum(dim=1, keepdim=True) + 1e-8)
        return self.norm(pooled)


class RecipeMoEEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        quantity_dim: int,
        num_experts: int,
        gate_hidden: int,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        self.num_experts = num_experts
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.quantity_proj = nn.Linear(1, quantity_dim)
        self.input_proj = nn.Linear(embed_dim + quantity_dim, embed_dim)
        self.experts = nn.ModuleList(
            [
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=embed_dim * 4,
                        dropout=dropout,
                        batch_first=True,
                    ),
                    num_layers=num_layers,
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, num_experts),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: torch.Tensor, quantities: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        tok = self.token_embed(token_ids)
        qty = self.quantity_proj(quantities.unsqueeze(-1))
        x = torch.cat([tok, qty], dim=-1) # [B, L, E+Q]
        x = self.input_proj(x) # [B, L, D]
        key_padding_mask = attn_mask == 0
        pooled_in = (x * attn_mask.unsqueeze(-1)).sum(dim=1) / (attn_mask.sum(dim=1, keepdim=True) + 1e-8)
        gate_logits = self.gate(pooled_in)
        gate_weights = F.softmax(gate_logits, dim=-1) # Linear -> ReLU -> Linear -> Softmax

        expert_outs = []
        for expert in self.experts:
            h = expert(x, src_key_padding_mask=key_padding_mask)
            masked = h * attn_mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / (attn_mask.sum(dim=1, keepdim=True) + 1e-8)
            expert_outs.append(pooled)
        stacked = torch.stack(expert_outs, dim=1)  # [B, E, D]
        mixed = torch.sum(stacked * gate_weights.unsqueeze(-1), dim=1)
        return self.norm(mixed)


class SustainabilityHead(nn.Module):
    def __init__(self, embed_dim: int, num_tasks: int = 4, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(32, embed_dim // 2)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_tasks)
            ]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        outputs = [head(z) for head in self.heads]
        return torch.cat(outputs, dim=-1)


def task_directions_from_names(task_names):
    direction_map = {
        "price_total": 1.0,
        "welfare": 1.0,
        "NutriScore_Scaled": -1.0,
        "Tot_env_100g": 1.0,
    }
    return [direction_map.get(name, 1.0) for name in task_names]


def multitask_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "rmse",
    task_weights=None,
    task_weight_mode: str = "static",
    task_log_vars: torch.Tensor = None,
) -> torch.Tensor:
    num_tasks = pred.shape[-1]
    if task_weight_mode not in {"static", "learned"}:
        raise ValueError(f"Unsupported task_weight_mode: {task_weight_mode}")
    if task_weight_mode == "learned":
        if task_log_vars is None:
            raise ValueError("task_log_vars is required when task_weight_mode='learned'")
        if task_log_vars.numel() != num_tasks:
            raise ValueError(f"task_log_vars length {task_log_vars.numel()} != num_tasks {num_tasks}")
        if loss_type == "mae":
            per_task = torch.mean(torch.abs(pred - target), dim=0)
        elif loss_type == "rmse":
            per_task = torch.mean((pred - target) ** 2, dim=0)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        precision = torch.exp(-task_log_vars)
        return torch.sum(precision * per_task + task_log_vars)

    if task_weights is None:
        weights = torch.ones(num_tasks, device=pred.device, dtype=pred.dtype)
    else:
        if not torch.is_tensor(task_weights):
            weights = torch.tensor(task_weights, device=pred.device, dtype=pred.dtype)
        else:
            weights = task_weights.to(device=pred.device, dtype=pred.dtype)
        if weights.numel() != num_tasks:
            raise ValueError(f"task_weights length {weights.numel()} != num_tasks {num_tasks}")
    weight_sum = weights.sum()
    if weight_sum <= 0:
        weights = torch.ones_like(weights)
        weight_sum = weights.sum()
    weights = weights / weight_sum

    if loss_type == "mae":
        per_task = torch.mean(torch.abs(pred - target), dim=0)
        return torch.sum(per_task * weights)
    if loss_type == "rmse":
        per_task_mse = torch.mean((pred - target) ** 2, dim=0)
        weighted_mse = torch.sum(per_task_mse * weights)
        return torch.sqrt(weighted_mse + 1e-8)
    raise ValueError(f"Unsupported loss_type: {loss_type}")

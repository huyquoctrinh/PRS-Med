from __future__ import annotations

from typing import Optional

import torch

from prs_med.configs.base import OptimizerConfig, SchedulerConfig


def create_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    if config.name.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=tuple(config.betas),
            weight_decay=config.weight_decay,
            eps=config.eps,
        )
    raise ValueError(f"Unsupported optimizer: {config.name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if config.name == "none":
        return None
    if config.name == "cosine":
        t_max = config.t_max or total_steps
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=config.eta_min)
    raise ValueError(f"Unsupported scheduler: {config.name}")



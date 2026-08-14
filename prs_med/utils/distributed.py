from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from prs_med.configs.base import DDPConfig, LoggingConfig


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return (not is_distributed()) or dist.get_rank() == 0


def ddp_bar(*args, **kwargs):
    return tqdm(*args, disable=not is_main_process(), **kwargs)


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def setup_logging(logging_config: LoggingConfig) -> None:
    log_path = logging_config.log_path()
    logging.basicConfig(
        filename=str(log_path) if is_main_process() else None,
        level=getattr(logging, logging_config.level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def init_distributed(ddp_config: DDPConfig) -> torch.device:
    dist.init_process_group(backend=ddp_config.backend, init_method=ddp_config.init_method)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return device


def wrap_ddp(model: torch.nn.Module, device: torch.device, ddp_config: DDPConfig) -> DDP:
    model = model.to(device)
    return DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
        find_unused_parameters=ddp_config.find_unused_parameters,
    )


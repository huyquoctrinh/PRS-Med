from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


@dataclass
class SchedulerConfig:
    """Configuration for learning-rate schedulers."""

    name: Literal["cosine", "none"] = "cosine"
    t_max: Optional[int] = None
    eta_min: float = 1e-6


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""

    name: Literal["adamw"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    eps: float = 1e-6
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class ModelConfig:
    """Configuration governing model construction."""

    model_path: str
    sam_checkpoint: str = ""
    save_dir: str = "training_results"
    base_model: Optional[str] = None
    device: str = "cuda"
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    version: str = "v1"
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    lora_modules_to_save: Optional[List[str]] = None
    load_8bit: bool = False
    load_4bit: bool = False
    sam_model_type: str = "vit_t"
    sam_model_id: str = "wanglab/medsam-vit-base"
    mask_decoder_variant: str = "v5"
    num_classes: int = 6
    freeze_image_encoder: bool = False

    def ensure_paths(self) -> None:
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Dataset and dataloader configuration."""

    data_path: str
    annotation_path: str = ""
    data_format: str = "csv"
    dialogues_dir: Optional[str] = None
    dialogues_prefix: str = "dialogues"
    batch_size: int = 4
    val_batch_size: Optional[int] = None
    num_workers: int = 4
    max_prompt_length: int = 512
    ignore_index: int = 0
    train_split: float = 0.9
    persistent_workers: bool = True
    pin_memory: bool = True

    def effective_val_batch_size(self) -> int:
        return self.val_batch_size or self.batch_size


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    epochs: int = 20
    gradient_clip_norm: float = 1.0
    log_interval: int = 10
    eval_interval: int = 1
    save_every: int = 1
    grad_accum_steps: int = 1
    seg_weight: float = 0.5
    lm_weight: float = 0.5
    cls_weight: float = 1.0
    warmup_steps: int = 200
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class DDPConfig:
    """Distributed training parameters."""

    backend: Literal["nccl", "gloo", "mpi"] = "nccl"
    init_method: Optional[str] = None
    seed: int = 42
    find_unused_parameters: bool = True


@dataclass
class LoggingConfig:
    """Logging destinations and verbosity."""

    log_dir: str = "logs"
    log_file: str = "training.log"
    level: Literal["INFO", "DEBUG"] = "INFO"

    def log_path(self) -> Path:
        path = Path(self.log_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path / self.log_file


@dataclass
class CheckpointConfig:
    """Checkpoint handling for training and inference."""

    output_dir: str
    resume_from: Optional[str] = None
    keep_last: int = 3

    def ensure_output_dir(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class ServingConfig:
    """Configuration for the Gradio serving UI."""

    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False
    max_queue: int = 16


@dataclass
class AppConfig:
    """Top-level application configuration."""

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DDPConfig = field(default_factory=DDPConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: Optional[CheckpointConfig] = None
    serving: Optional[ServingConfig] = None

    def resolve(self) -> "AppConfig":
        self.model.ensure_paths()
        if self.checkpoint:
            self.checkpoint.ensure_output_dir()
        return self



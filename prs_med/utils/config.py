from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from prs_med.configs.base import AppConfig, CheckpointConfig, DataConfig, DDPConfig, LoggingConfig, ModelConfig, ServingConfig, TrainingConfig


def _build_model(cfg: Dict[str, Any]) -> ModelConfig:
    return ModelConfig(**cfg)


def _build_data(cfg: Dict[str, Any]) -> DataConfig:
    return DataConfig(**cfg)


def _build_training(cfg: Optional[Dict[str, Any]]) -> TrainingConfig:
    if cfg is None:
        return TrainingConfig()
    from prs_med.configs.base import OptimizerConfig, SchedulerConfig
    cfg = dict(cfg)
    if "optimizer" in cfg and isinstance(cfg["optimizer"], dict):
        cfg["optimizer"] = OptimizerConfig(**cfg["optimizer"])
    if "scheduler" in cfg and isinstance(cfg["scheduler"], dict):
        cfg["scheduler"] = SchedulerConfig(**cfg["scheduler"])
    return TrainingConfig(**cfg)


def _build_serving(cfg: Optional[Dict[str, Any]]) -> Optional[ServingConfig]:
    if cfg is None:
        return None
    return ServingConfig(**cfg)


def _build_ddp(cfg: Optional[Dict[str, Any]]) -> DDPConfig:
    if cfg is None:
        return DDPConfig()
    return DDPConfig(**cfg)


def _build_logging(cfg: Optional[Dict[str, Any]]) -> LoggingConfig:
    if cfg is None:
        return LoggingConfig()
    return LoggingConfig(**cfg)


def _build_checkpoint(cfg: Optional[Dict[str, Any]]) -> Optional[CheckpointConfig]:
    if cfg is None:
        return None
    return CheckpointConfig(**cfg)


def load_config(path: str) -> AppConfig:
    """Load AppConfig from YAML or JSON file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix in {".yaml", ".yml"}:
            raw_cfg = yaml.safe_load(handle)
        else:
            raw_cfg = json.load(handle)

    model_cfg = _build_model(raw_cfg["model"])
    data_cfg = _build_data(raw_cfg["data"])
    training_cfg = _build_training(raw_cfg.get("training"))
    ddp_cfg = _build_ddp(raw_cfg.get("distributed"))
    logging_cfg = _build_logging(raw_cfg.get("logging"))
    checkpoint_cfg = _build_checkpoint(raw_cfg.get("checkpoint"))
    serving_cfg = _build_serving(raw_cfg.get("serving"))

    return AppConfig(
        model=model_cfg,
        data=data_cfg,
        training=training_cfg,
        distributed=ddp_cfg,
        logging=logging_cfg,
        checkpoint=checkpoint_cfg,
        serving=serving_cfg,
    ).resolve()


def parse_args(default_config: Optional[str] = None) -> AppConfig:
    """Parse CLI arguments to obtain a config path and load it."""
    parser = argparse.ArgumentParser(description="PRS-Med training entrypoint.")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to YAML/JSON configuration file.",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help="Optional key=value overrides applied after loading the config.",
    )
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("A configuration file path must be provided via --config.")

    app_config = load_config(args.config)
    if not args.override:
        return app_config

    overrides = dict(item.split("=", maxsplit=1) for item in args.override)

    for key, value in overrides.items():
        _apply_override(app_config, key, value)

    return app_config.resolve()


def _apply_override(app_config: AppConfig, key: str, value: str) -> None:
    """Apply a dotted-path override to the AppConfig."""
    sections = key.split(".")
    target = app_config
    for attr in sections[:-1]:
        if not hasattr(target, attr):
            raise AttributeError(f"Unknown config attribute: {attr} in {key}")
        target = getattr(target, attr)

    final_attr = sections[-1]
    if not hasattr(target, final_attr):
        raise AttributeError(f"Unknown config attribute: {final_attr} in {key}")

    current_value = getattr(target, final_attr)
    cast_value = _cast_value(value, type(current_value))
    setattr(target, final_attr, cast_value)


def _cast_value(value: str, expected_type: Any) -> Any:
    """Attempt to cast string overrides to the expected type."""
    if expected_type is bool:
        return value.lower() in {"1", "true", "yes"}
    if expected_type is int:
        return int(value)
    if expected_type is float:
        return float(value)
    return value



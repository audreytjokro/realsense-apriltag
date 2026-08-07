"""Training/evaluation orchestration for the BiGRU leave-one-session-out area experiment.

Everything data-, loss-, and metric-related is imported unchanged from
analysis.spatial_sequence (build_prepared_data, the Dataset classes, masked_loss via
train_one_epoch/validation_loss, evaluate_and_save, checkpoint_selection_metric,
parameter_groups, seeding/checkpoint utilities). Only the model class and the small
amount of glue needed to plug a different nn.Module into that same loop are new.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from analysis.spatial_sequence.core import (
    EXPERIMENT_ROOT,
    REFERENCE_SEQUENCE_LENGTH,
    REPOSITORY_ROOT,
    PreparedData,
    SpatialAnchorDataset,
    SpatialWindowDataset,
    build_prepared_data,
)
from analysis.spatial_sequence.evaluation import evaluate_and_save, validation_loss
from analysis.spatial_sequence.metrics import checkpoint_selection_metric, selection_is_better
from analysis.spatial_sequence.models import parameter_groups
from analysis.spatial_sequence.training import (
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)

from .models import (
    GRU_HIDDEN_SIZE,
    GRU_LAYERS,
    GRU_V2_DROPOUT,
    GRU_V2_HIDDEN_SIZE,
    BiGRUAreaModel,
    BiGRUAreaModelV2,
    count_parameters,
)

TARGET = "area"
TEMPORAL_MODE = "bidirectional"
ARCHITECTURE = "bigru"
ARCHITECTURE_VERSIONS = ("v1", "v2")


@dataclass(frozen=True)
class RunConfig:
    held_out_session: str
    seed: int = 0
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    minimum_learning_rate: float = 1e-6
    gradient_clip: float = 1.0
    device: str = "auto"
    output_directory: str | None = None
    experiment_root: str = str(EXPERIMENT_ROOT)
    architecture_version: str = "v1"

    def validate(self) -> None:
        if not self.held_out_session:
            raise ValueError("held_out_session is required; this experiment is LOSO-only")
        if self.architecture_version not in ARCHITECTURE_VERSIONS:
            raise ValueError(f"Unknown architecture_version: {self.architecture_version}")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("Epochs and batch size must be positive")
        if self.learning_rate <= 0 or self.minimum_learning_rate < 0:
            raise ValueError("Learning rates must be non-negative, with a positive initial rate")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("Minimum learning rate cannot exceed the initial learning rate")
        if self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Weight decay must be non-negative and gradient clip positive")


def build_model(config: RunConfig) -> BiGRUAreaModel | BiGRUAreaModelV2:
    if config.architecture_version == "v1":
        return BiGRUAreaModel()
    return BiGRUAreaModelV2()


def fixed_topology(architecture_version: str = "v1") -> dict[str, Any]:
    if architecture_version == "v1":
        return {
            "architecture": ARCHITECTURE,
            "architecture_version": "v1",
            "input_channels": 32,
            "sequence_length": REFERENCE_SEQUENCE_LENGTH,
            "d_model": 96,
            "gru_hidden_size": GRU_HIDDEN_SIZE,
            "gru_layers": GRU_LAYERS,
            "gru_bidirectional": True,
            "output_norm": None,
            "output_dropout": 0.0,
            "temporal_pooling": None,
            "extra_recurrent_layers": 0,
            "head": "linear",
            "dense_supervision": True,
            "anchor_evaluation": True,
            "bidirectional_anchor_index": REFERENCE_SEQUENCE_LENGTH // 2,
        }
    return {
        "architecture": ARCHITECTURE,
        "architecture_version": "v2",
        "input_channels": 32,
        "sequence_length": REFERENCE_SEQUENCE_LENGTH,
        "d_model": 96,
        "gru_hidden_size": GRU_V2_HIDDEN_SIZE,
        "gru_layers": GRU_LAYERS,
        "gru_bidirectional": True,
        "output_norm": "layer_norm",
        "output_dropout": GRU_V2_DROPOUT,
        "temporal_pooling": None,
        "extra_recurrent_layers": 0,
        "head": "linear",
        "dense_supervision": True,
        "anchor_evaluation": True,
        "bidirectional_anchor_index": REFERENCE_SEQUENCE_LENGTH // 2,
    }


def default_run_directory(config: RunConfig) -> Path:
    run_name = f"bigru_bidirectional_seed{config.seed}"
    return (
        REPOSITORY_ROOT
        / "output"
        / "gru-sequence"
        / config.architecture_version
        / "leave-one-session-out"
        / "runs"
        / TARGET
        / config.held_out_session
        / run_name
    )


def _signature(config: RunConfig, prepared: PreparedData) -> dict[str, Any]:
    configuration = asdict(config)
    configuration.pop("device")
    configuration.pop("output_directory")
    return {
        "configuration": configuration,
        "fixed_topology": fixed_topology(config.architecture_version),
        "source_hashes": prepared.source_hashes,
        "normalization": prepared.stats.as_dict(),
        "session_ids": [session.info.session_id for session in prepared.sessions],
        "session_slugs": [session.info.slug for session in prepared.sessions],
        "split": "four-full-training-sessions-one-full-validation-session",
        "normalization_method": "unique-training-physical-rows",
        "checkpoint_selection": "task_metric_then_anchor_loss_then_epoch",
    }


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def ensure_output_directory(config: RunConfig) -> Path:
    output_directory = (
        Path(config.output_directory).expanduser().resolve()
        if config.output_directory
        else default_run_directory(config)
    )
    if output_directory.exists():
        if list(output_directory.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_directory}")
    else:
        output_directory.mkdir(parents=True)
    return output_directory


def train_run(config: RunConfig) -> Path:
    """Train one (held_out_session, seed) LOSO fold. No resume support (deliberately
    out of scope for this single controlled experiment; runs are short enough not to
    need it -- see README for why this differs from spatial_sequence.training.train_run).
    """
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    prepared = build_prepared_data(
        TARGET,
        experiment_root=Path(config.experiment_root),
        evaluation_scheme="leave-one-session-out",
        held_out_session=config.held_out_session,
    )
    validation_sessions = [
        item.info.slug
        for item in prepared.sessions
        if any(block.split == "validation" for block in item.blocks)
    ]
    if len(validation_sessions) != 1 or validation_sessions[0] != config.held_out_session:
        raise ValueError("LOSO preparation did not isolate the requested held-out session")

    signature = _signature(config, prepared)
    output_directory = ensure_output_directory(config)
    _save_json(output_directory / "resolved_config.json", asdict(config))
    _save_json(output_directory / "data_signature.json", signature)

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        SpatialWindowDataset(prepared, "train"),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        SpatialWindowDataset(prepared, "validation"),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    if len(train_loader) == 0:
        raise ValueError("Training has no complete batches; reduce batch_size for this dataset")
    anchor_loader = DataLoader(
        SpatialAnchorDataset(prepared, TEMPORAL_MODE),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = build_model(config).to(device)
    optimizer = AdamW(parameter_groups(model, config.weight_decay), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.minimum_learning_rate)

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_selection_value: float | None = None
    selection_metric_name: str | None = None
    selection_direction: str | None = None

    for epoch in range(1, config.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss, train_count = train_one_epoch(
            model, train_loader, optimizer, device, TARGET, config.gradient_clip,
        )
        dense_validation_loss, dense_validation_count = validation_loss(
            model, validation_loader, device, TARGET,
        )
        anchor_metrics, _ = evaluate_and_save(model, anchor_loader, device, prepared, None)
        anchor_loss = float(anchor_metrics["pooled_validation_loss"])
        metric_name, metric_value, metric_direction = checkpoint_selection_metric(
            TARGET, anchor_metrics,
        )
        if selection_metric_name is not None and (
            metric_name != selection_metric_name or metric_direction != selection_direction
        ):
            raise ValueError("Checkpoint selection criterion changed during the run")
        selection_metric_name = metric_name
        selection_direction = metric_direction
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "dense_validation_loss": dense_validation_loss,
                "anchor_validation_loss": anchor_loss,
                "selection_metric_name": metric_name,
                "selection_metric_value": metric_value,
                "selection_direction": metric_direction,
                "train_valid_elements": train_count,
                "dense_validation_valid_elements": dense_validation_count,
            }
        )
        scheduler.step()
        improved = selection_is_better(
            metric_value, anchor_loss, epoch,
            best_selection_value, best_loss, best_epoch, metric_direction,
        )
        if improved:
            best_loss = anchor_loss
            best_epoch = epoch
            best_selection_value = metric_value
            save_checkpoint(
                output_directory / "best.pt",
                {
                    "format_version": 1,
                    "config": asdict(config),
                    "signature": signature,
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "best_selection_value": best_selection_value,
                    "selection_metric_name": selection_metric_name,
                    "selection_direction": selection_direction,
                },
            )

    _write_history(output_directory / "history.csv", history)
    if not (output_directory / "best.pt").is_file():
        raise RuntimeError("Training did not create a best checkpoint")
    evaluate_checkpoint(output_directory / "best.pt", output_directory=output_directory)
    return output_directory


def evaluate_checkpoint(checkpoint_path: Path, output_directory: Path | None = None) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    device = resolve_device("auto")
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = RunConfig(**dict(checkpoint["config"]))
    prepared = build_prepared_data(
        TARGET,
        experiment_root=Path(config.experiment_root),
        evaluation_scheme="leave-one-session-out",
        held_out_session=config.held_out_session,
    )
    if checkpoint["signature"] != _signature(config, prepared):
        raise ValueError("Checkpoint configuration or data signature is incompatible")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    anchor_loader = DataLoader(
        SpatialAnchorDataset(prepared, TEMPORAL_MODE),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    destination = output_directory or checkpoint_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    metrics, aggregate = evaluate_and_save(model, anchor_loader, device, prepared, destination)
    metrics["best_epoch"] = int(checkpoint["best_epoch"])
    metrics["selection_metric_name"] = checkpoint.get("selection_metric_name")
    metrics["selection_direction"] = checkpoint.get("selection_direction")
    metrics["best_selection_value"] = checkpoint.get("best_selection_value")
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["exploratory_validation_only"] = True
    metrics["evaluation_scheme"] = "leave-one-session-out"
    metrics["held_out_session"] = config.held_out_session
    metrics["seed"] = config.seed
    metrics["parameter_count"] = count_parameters(model)
    _save_json(destination / "metrics.json", metrics)
    return metrics

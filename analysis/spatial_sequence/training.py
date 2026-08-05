from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .core import (
    EVALUATION_SCHEMES,
    EXPERIMENT_ROOT,
    POSITION_CLASS_COUNT,
    POSITION_INTERIOR_BINS,
    REFERENCE_SEQUENCE_LENGTH,
    REPOSITORY_ROOT,
    PreparedData,
    SpatialAnchorDataset,
    SpatialWindowDataset,
    anchor_records_for_sessions,
    build_prepared_data,
)
from .evaluation import evaluate_and_save, masked_loss, validation_loss
from .metrics import checkpoint_selection_metric, selection_is_better
from .models import (
    BLOCK_COUNT,
    CNN_DILATIONS,
    CNN_KERNEL_SIZE,
    D_MODEL,
    DROPOUT,
    FFN_DIM,
    HEAD_COUNT,
    SpatialSequenceModel,
    parameter_groups,
)


@dataclass(frozen=True)
class RunConfig:
    architecture: str
    temporal_mode: str
    target: str
    session: str | None = None
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
    sequence_length: int = REFERENCE_SEQUENCE_LENGTH
    evaluation_scheme: str = "within-session"
    held_out_session: str | None = None

    def validate(self) -> None:
        if self.architecture not in {"transformer", "temporal-cnn"}:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        if self.temporal_mode not in {"causal", "bidirectional"}:
            raise ValueError(f"Unknown temporal mode: {self.temporal_mode}")
        if self.target not in {"position", "distance", "area", "height", "velocity"}:
            raise ValueError(f"Unknown target: {self.target}")
        if self.target == "position" and not self.session:
            raise ValueError("Position runs require a session selector")
        if self.target != "position" and self.session is not None:
            raise ValueError("Only position runs accept a session selector")
        if self.sequence_length not in {6, 12, 18, REFERENCE_SEQUENCE_LENGTH}:
            raise ValueError("Run sequence length must be one of 6, 12, 18, or 24")
        if self.evaluation_scheme not in EVALUATION_SCHEMES:
            raise ValueError(f"Unknown evaluation scheme: {self.evaluation_scheme}")
        if self.evaluation_scheme == "within-session":
            if self.held_out_session is not None:
                raise ValueError("Within-session runs do not accept --held-out-session")
        else:
            if self.held_out_session is None:
                raise ValueError("LOSO runs require --held-out-session")
            if self.target not in {"area", "distance"}:
                raise ValueError("LOSO runs are defined only for area and distance")
            if self.temporal_mode != "bidirectional":
                raise ValueError("LOSO runs use bidirectional context")
            if self.sequence_length != REFERENCE_SEQUENCE_LENGTH:
                raise ValueError("LOSO runs use the canonical sequence length")
        if self.sequence_length != REFERENCE_SEQUENCE_LENGTH and (
            self.architecture != "temporal-cnn"
            or self.temporal_mode != "bidirectional"
            or self.target not in {"area", "distance"}
        ):
            raise ValueError(
                "Sequence-length ablations are bidirectional Temporal CNN area/distance runs"
            )
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("Epochs and batch size must be positive")
        if self.learning_rate <= 0 or self.minimum_learning_rate < 0:
            raise ValueError("Learning rates must be non-negative, with a positive initial rate")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("Minimum learning rate cannot exceed the initial learning rate")
        if self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Weight decay must be non-negative and gradient clip positive")


def fixed_topology(
    sequence_length: int = REFERENCE_SEQUENCE_LENGTH,
) -> dict[str, Any]:
    return {
        "input_channels": 32,
        "sequence_length": sequence_length,
        "d_model": D_MODEL,
        "blocks": BLOCK_COUNT,
        "ffn_dim": FFN_DIM,
        "attention_heads": HEAD_COUNT,
        "cnn_kernel_size": CNN_KERNEL_SIZE,
        "cnn_dilations": list(CNN_DILATIONS),
        "dropout": DROPOUT,
        "pre_norm": True,
        "dense_supervision": True,
        "position_head": "joint-paper-only",
        "position_interior_bins": POSITION_INTERIOR_BINS,
        "position_class_count": POSITION_CLASS_COUNT,
        "anchor_evaluation": True,
        "causal_anchor_index": sequence_length - 1,
        "bidirectional_anchor_index": sequence_length // 2,
    }


def default_run_directory(config: RunConfig) -> Path:
    run_name = f"{config.architecture}_{config.temporal_mode}_seed{config.seed}"
    output_root = REPOSITORY_ROOT / "output" / "spatial-sequence"
    if config.evaluation_scheme == "leave-one-session-out":
        return (
            output_root
            / "leave-one-session-out"
            / "runs"
            / config.target
            / str(config.held_out_session)
            / run_name
        )
    if config.sequence_length != REFERENCE_SEQUENCE_LENGTH:
        return (
            output_root
            / "sequence-length"
            / "runs"
            / config.target
            / f"length-{config.sequence_length}"
            / run_name
        )
    base = output_root / "runs" / config.target
    if config.target == "position":
        return base / str(config.session) / run_name
    return base / run_name


def position_video_path(config: RunConfig) -> Path:
    if config.target != "position" or config.session is None:
        raise ValueError("Position video paths require a position run")
    model = "tcnn" if config.architecture == "temporal-cnn" else "transformer"
    filename = (
        f"position_{config.session}_{config.temporal_mode}_{model}_seed{config.seed}.mp4"
    )
    return REPOSITORY_ROOT / "output" / "spatial-sequence" / "videos" / filename


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _signature(config: RunConfig, prepared: PreparedData) -> dict[str, Any]:
    configuration = asdict(config)
    configuration.pop("device")
    configuration.pop("output_directory")
    return {
        "configuration": configuration,
        "fixed_topology": fixed_topology(config.sequence_length),
        "source_hashes": prepared.source_hashes,
        "normalization": prepared.stats.as_dict(),
        "session_ids": [session.info.session_id for session in prepared.sessions],
        "session_slugs": [
            session.info.slug for session in prepared.sessions
        ],
        "split": (
            "train-guard-validation-guard-train"
            if config.evaluation_scheme == "within-session"
            else "four-full-training-sessions-one-full-validation-session"
        ),
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


def _split_summary(prepared: PreparedData, temporal_mode: str) -> dict[str, Any]:
    sessions: dict[str, Any] = {}
    for session in prepared.sessions:
        session_key = session.info.slug
        sessions[session_key] = {
            "session_id": session.info.session_id,
            "session_slug": session.info.slug,
            "active_rows": int(len(session.sensors)),
            "pose_valid_rows": int(np.sum(session.pose_mask)),
            "blocks": [
                {
                    "split": block.split,
                    "fragment": block.fragment,
                    "active_start_inclusive": block.start,
                    "active_stop_exclusive": block.stop,
                    "row_count": block.stop - block.start,
                }
                for block in session.blocks
            ],
        }
    return {
        "sequence_length": prepared.sequence_length,
        "guard_length": 23 if prepared.evaluation_scheme == "within-session" else 0,
        "train_window_count": len(prepared.train_windows),
        "validation_window_count": len(prepared.validation_windows),
        "anchor_validation_count": len(
            anchor_records_for_sessions(
                prepared.sessions,
                temporal_mode,
                prepared.sequence_length,
                full_validation_block=(
                    prepared.evaluation_scheme == "leave-one-session-out"
                ),
            )
        ),
        "evaluation_scheme": prepared.evaluation_scheme,
        "normalization": "unique-training-physical-rows",
        "sessions": sessions,
    }


def ensure_output_directory(
    config: RunConfig,
    resume_checkpoint: Path | None,
) -> Path:
    if config.output_directory:
        output_directory = Path(config.output_directory).expanduser().resolve()
    elif resume_checkpoint is not None:
        output_directory = resume_checkpoint.expanduser().resolve().parent
    else:
        output_directory = default_run_directory(config)
    if output_directory.exists():
        entries = list(output_directory.iterdir())
        if entries and resume_checkpoint is None:
            raise FileExistsError(
                f"Non-resume output directory is not empty: {output_directory}"
            )
    else:
        output_directory.mkdir(parents=True)
    return output_directory


def capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_loader_generator": generator.get_state(),
    }


def restore_rng_state(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["data_loader_generator"])


def _checkpoint_payload(
    config: RunConfig,
    signature: dict[str, Any],
    model: SpatialSequenceModel,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    history: list[dict[str, Any]],
    best_loss: float,
    best_epoch: int,
    generator: torch.Generator,
    best_selection_value: float | None = None,
    selection_metric_name: str | None = None,
    selection_direction: str | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "config": asdict(config),
        "signature": signature,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "history": history,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "best_selection_value": best_selection_value,
        "selection_metric_name": selection_metric_name,
        "selection_direction": selection_direction,
        "rng_state": capture_rng_state(generator),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def _payload_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and np.array_equal(left, right, equal_nan=True)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_payload_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_payload_values_equal(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)


def rewrite_checkpoint_metadata_atomic(
    path: Path,
    replacements: dict[str, Any],
) -> None:
    """Replace selected top-level metadata after a verified atomic rewrite."""
    original = load_checkpoint(path, torch.device("cpu"))
    unknown = set(replacements) - set(original)
    if unknown:
        raise KeyError(f"Checkpoint metadata keys do not exist: {sorted(unknown)}")
    rewritten = dict(original)
    rewritten.update(replacements)
    temporary = path.with_name(path.name + ".migrating")
    try:
        torch.save(rewritten, temporary)
        verified = load_checkpoint(temporary, torch.device("cpu"))
        if not _payload_values_equal(rewritten, verified):
            raise ValueError(f"Checkpoint rewrite verification failed: {path}")
        for key, value in original.items():
            if key not in replacements and not _payload_values_equal(value, verified[key]):
                raise ValueError(
                    f"Checkpoint training state changed while rewriting {path}: {key}"
                )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _create_loaders(
    prepared: PreparedData,
    config: RunConfig,
    generator: torch.Generator,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = SpatialWindowDataset(prepared, "train")
    validation_dataset = SpatialWindowDataset(prepared, "validation")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "Training has no complete batches; reduce --batch-size for this dataset"
        )
    anchor_dataset = SpatialAnchorDataset(prepared, config.temporal_mode)
    anchor_loader = DataLoader(
        anchor_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return train_loader, validation_loader, anchor_loader


def train_one_epoch(
    model: SpatialSequenceModel,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    target_name: str,
    gradient_clip: float,
) -> tuple[float, int]:
    model.train()
    loss_sum = 0.0
    valid_total = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, valid_count = masked_loss(target_name, model(inputs), target, mask)
        if valid_count == 0:
            continue
        normalized_loss = loss / valid_count
        normalized_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        loss_sum += float(loss.item())
        valid_total += valid_count
    if valid_total == 0:
        raise ValueError("Training epoch contains no valid labels")
    return loss_sum / valid_total, valid_total


def train_run(
    config: RunConfig,
    resume_checkpoint: Path | None = None,
    render_position_video: bool = False,
) -> Path:
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    prepared = build_prepared_data(
        config.target,
        session=config.session,
        experiment_root=Path(config.experiment_root),
        sequence_length=config.sequence_length,
        evaluation_scheme=config.evaluation_scheme,
        held_out_session=config.held_out_session,
    )
    if config.target == "position":
        canonical_session = prepared.sessions[0].info.slug
        if config.session != canonical_session:
            config = replace(config, session=canonical_session)
    if config.evaluation_scheme == "leave-one-session-out":
        validation_sessions = [
            item.info.slug
            for item in prepared.sessions
            if any(block.split == "validation" for block in item.blocks)
        ]
        if len(validation_sessions) != 1:
            raise ValueError("LOSO preparation must contain one validation session")
        if config.held_out_session != validation_sessions[0]:
            config = replace(config, held_out_session=validation_sessions[0])
    signature = _signature(config, prepared)
    resume_payload: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        resume_payload = load_checkpoint(resume_checkpoint, device)
        if resume_payload["signature"] != signature:
            raise ValueError("Checkpoint configuration or data signature is incompatible")
    output_directory = ensure_output_directory(config, resume_checkpoint)
    _save_json(output_directory / "resolved_config.json", asdict(config))
    _save_json(output_directory / "data_signature.json", signature)
    _save_json(
        output_directory / "split_summary.json",
        _split_summary(prepared, config.temporal_mode),
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader, validation_loader, anchor_loader = _create_loaders(
        prepared, config, generator
    )
    model = SpatialSequenceModel(
        config.architecture,
        config.temporal_mode,
        config.target,
        maximum_sequence_length=config.sequence_length,
    ).to(device)
    optimizer = AdamW(
        parameter_groups(model, config.weight_decay),
        lr=config.learning_rate,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.minimum_learning_rate,
    )

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_selection_value: float | None = None
    selection_metric_name: str | None = None
    selection_direction: str | None = None
    start_epoch = 1
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        history = list(resume_payload["history"])
        best_loss = float(resume_payload["best_loss"])
        best_epoch = int(resume_payload["best_epoch"])
        best_selection_value = resume_payload.get("best_selection_value")
        selection_metric_name = resume_payload.get("selection_metric_name")
        selection_direction = resume_payload.get("selection_direction")
        start_epoch = int(resume_payload["epoch"]) + 1
        restore_rng_state(resume_payload["rng_state"], generator)

    for epoch in range(start_epoch, config.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss, train_count = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config.target,
            config.gradient_clip,
        )
        dense_validation_loss, dense_validation_count = validation_loss(
            model,
            validation_loader,
            device,
            config.target,
        )
        anchor_metrics, _ = evaluate_and_save(
            model,
            anchor_loader,
            device,
            prepared,
            None,
        )
        anchor_loss = float(anchor_metrics["pooled_validation_loss"])
        metric_name, metric_value, metric_direction = checkpoint_selection_metric(
            config.target,
            anchor_metrics,
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
                "anchor_metrics_json": json.dumps(anchor_metrics["pooled"], sort_keys=True),
                "train_valid_elements": train_count,
                "dense_validation_valid_elements": dense_validation_count,
            }
        )
        scheduler.step()
        improved = selection_is_better(
            metric_value,
            anchor_loss,
            epoch,
            best_selection_value,
            best_loss,
            best_epoch,
            metric_direction,
        )
        if improved:
            best_loss = anchor_loss
            best_epoch = epoch
            best_selection_value = metric_value
        payload = _checkpoint_payload(
            config,
            signature,
            model,
            optimizer,
            scheduler,
            epoch,
            history,
            best_loss,
            best_epoch,
            generator,
            best_selection_value=best_selection_value,
            selection_metric_name=selection_metric_name,
            selection_direction=selection_direction,
        )
        if improved:
            save_checkpoint(output_directory / "best.pt", payload)
        if epoch % 5 == 0:
            save_checkpoint(output_directory / f"epoch_{epoch:03d}.pt", payload)
        if epoch == config.epochs:
            save_checkpoint(output_directory / "final.pt", payload)
        _write_history(output_directory / "history.csv", history)

    if not (output_directory / "best.pt").is_file():
        raise RuntimeError("Training did not create a best checkpoint")
    evaluate_checkpoint(
        output_directory / "best.pt",
        output_directory=output_directory,
        render_video=render_position_video,
    )
    return output_directory


def evaluate_checkpoint(
    checkpoint_path: Path,
    output_directory: Path | None = None,
    selector_overrides: dict[str, str | None] | None = None,
    write_metrics: bool = True,
    render_video: bool | None = None,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_config = dict(raw["config"])
    config = RunConfig(**raw_config)
    if selector_overrides:
        for key in ("architecture", "temporal_mode", "target"):
            supplied = selector_overrides.get(key)
            if supplied is not None and supplied != getattr(config, key):
                raise ValueError(
                    f"Checkpoint {key}={getattr(config, key)!r} conflicts with {supplied!r}"
                )
    device = resolve_device(config.device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    prepared = build_prepared_data(
        config.target,
        session=config.session,
        experiment_root=Path(config.experiment_root),
        sequence_length=config.sequence_length,
        evaluation_scheme=config.evaluation_scheme,
        held_out_session=config.held_out_session,
    )
    if selector_overrides and selector_overrides.get("session") is not None:
        supplied_session = str(selector_overrides["session"])
        info = prepared.sessions[0].info
        aliases = {
            info.session_id,
            info.trial_id,
            info.slug,
            info.session_directory.name,
        }
        if supplied_session not in aliases:
            raise ValueError(
                f"Checkpoint session={config.session!r} conflicts with {supplied_session!r}"
            )
    if checkpoint["signature"] != _signature(config, prepared):
        raise ValueError("Checkpoint configuration or data signature is incompatible")
    model = SpatialSequenceModel(
        config.architecture,
        config.temporal_mode,
        config.target,
        maximum_sequence_length=config.sequence_length,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    _, _, anchor_loader = _create_loaders(
        prepared,
        config,
        torch.Generator().manual_seed(config.seed),
    )
    destination = output_directory or checkpoint_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    metrics, aggregate = evaluate_and_save(
        model, anchor_loader, device, prepared, destination
    )
    metrics["best_epoch"] = int(checkpoint["best_epoch"])
    metrics["selection_metric_name"] = checkpoint.get("selection_metric_name")
    metrics["selection_direction"] = checkpoint.get("selection_direction")
    metrics["best_selection_value"] = checkpoint.get("best_selection_value")
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["exploratory_validation_only"] = True
    metrics["evaluation_scheme"] = config.evaluation_scheme
    metrics["sequence_length"] = config.sequence_length
    metrics["held_out_session"] = config.held_out_session
    if write_metrics:
        _save_json(destination / "metrics.json", metrics)
    if render_video is None:
        render_video = False
    if config.target == "position" and render_video:
        from .visualization import render_position_video

        video_path = position_video_path(config)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        render_position_video(prepared, aggregate, video_path)
    if (
        config.evaluation_scheme == "within-session"
        and config.sequence_length == REFERENCE_SEQUENCE_LENGTH
    ):
        from .visualization import render_run_diagnostics

        render_run_diagnostics(
            prepared,
            aggregate,
            destination / "diagnostics.png",
            architecture=config.architecture,
            temporal_mode=config.temporal_mode,
            seed=config.seed,
        )
    return metrics

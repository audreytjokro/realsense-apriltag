from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from .core import (
    POSITION_INTERIOR_BINS,
    NormalizationStats,
    PreparedData,
    inverse_regression_target,
)
from .metrics import summarize_metrics


def masked_loss(
    target_name: str,
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if target_name == "position":
        valid = mask
        if not torch.any(valid):
            return outputs["position"].new_zeros(()), 0
        return (
            F.cross_entropy(
                outputs["position"][valid],
                target[valid].long(),
                reduction="sum",
            ),
            int(valid.sum().item()),
        )
    if target_name == "area":
        valid = mask
        if not torch.any(valid):
            return outputs["area"].new_zeros(()), 0
        return (
            F.cross_entropy(outputs["area"][valid], target[valid].long(), reduction="sum"),
            int(valid.sum().item()),
        )
    prediction = outputs[target_name]
    valid = mask
    if not torch.any(valid):
        return prediction.new_zeros(()), 0
    return (
        F.smooth_l1_loss(prediction[valid], target[valid], beta=1.0, reduction="sum"),
        int(valid.sum().item()),
    )


def validation_loss(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    target_name: str,
) -> tuple[float, int]:
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            loss, valid_count = masked_loss(target_name, model(inputs), target, mask)
            loss_sum += float(loss.item())
            count += valid_count
    if count == 0:
        raise ValueError("Validation contains no valid labels")
    return loss_sum / count, count


def _select_anchor_outputs(
    outputs: dict[str, torch.Tensor],
    offsets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch_indices = torch.arange(offsets.shape[0], device=offsets.device)
    return {
        name: values[batch_indices, offsets]
        for name, values in outputs.items()
    }


def collect_anchor_predictions(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    target_name: str,
    stats: NormalizationStats,
) -> tuple[dict[str, np.ndarray], float]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    loss_sum = 0.0
    loss_count = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            target_device = batch["target"].to(device)
            mask_device = batch["mask"].to(device)
            offsets = batch["target_offset"].to(device)
            outputs = _select_anchor_outputs(model(inputs), offsets)
            loss, count = masked_loss(target_name, outputs, target_device, mask_device)
            loss_sum += float(loss.item())
            loss_count += count
            collected["session_index"].append(batch["session_index"].numpy())
            collected["raw_row"].append(batch["raw_row_index"].numpy())
            collected["fragment"].append(batch["fragment"].numpy())
            collected["paper_xy_cm"].append(batch["paper_xy_cm"].numpy())
            target = batch["target"].numpy()
            mask = batch["mask"].numpy()
            if target_name == "position":
                collected["truth"].append(target.reshape(-1))
                collected["mask"].append(mask.reshape(-1))
                collected["joint_probabilities"].append(
                    torch.softmax(outputs["position"], dim=-1)
                    .cpu()
                    .numpy()
                    .reshape(-1, POSITION_INTERIOR_BINS, POSITION_INTERIOR_BINS)
                )
            elif target_name == "area":
                collected["truth"].append(target.reshape(-1))
                collected["mask"].append(mask.reshape(-1))
                collected["probabilities"].append(
                    torch.softmax(outputs["area"], dim=-1).cpu().numpy().reshape(-1, 3)
                )
            else:
                width = 1 if target_name == "height" else 2
                scaled_prediction = outputs[target_name].cpu().numpy().reshape(-1, width)
                scaled_truth = target.reshape(-1, width)
                collected["prediction"].append(
                    inverse_regression_target(target_name, scaled_prediction, stats)
                )
                collected["truth"].append(
                    inverse_regression_target(target_name, scaled_truth, stats)
                )
                collected["mask"].append(mask.reshape(-1, width))
    if loss_count == 0:
        raise ValueError("Anchor validation contains no valid labels")
    return (
        {name: np.concatenate(values, axis=0) for name, values in collected.items()},
        loss_sum / loss_count,
    )


def _mean_by_group(values: np.ndarray, inverse: np.ndarray, group_count: int) -> np.ndarray:
    values = np.asarray(values)
    result = np.zeros((group_count,) + values.shape[1:], dtype=np.float64)
    counts = np.bincount(inverse, minlength=group_count).astype(np.float64)
    np.add.at(result, inverse, values)
    reshape = (group_count,) + (1,) * (values.ndim - 1)
    return result / counts.reshape(reshape)


def aggregate_physical_tokens(
    occurrences: dict[str, np.ndarray],
    target_name: str,
) -> dict[str, np.ndarray]:
    keys = np.column_stack(
        [
            occurrences["session_index"],
            occurrences["raw_row"],
            occurrences["fragment"],
        ]
    )
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    group_count = len(unique_keys)
    counts = np.bincount(inverse, minlength=group_count).astype(np.int64)
    aggregate: dict[str, np.ndarray] = {
        "session_index": unique_keys[:, 0].astype(np.int64),
        "raw_row": unique_keys[:, 1].astype(np.int64),
        "fragment": unique_keys[:, 2].astype(np.int64),
        "occurrence_count": counts,
    }
    mask = occurrences["mask"]
    if mask.ndim == 1:
        mask = mask[:, None]
    aggregate_mask = np.zeros((group_count, mask.shape[1]), dtype=bool)
    truth_shape = () if occurrences["truth"].ndim == 1 else occurrences["truth"].shape[1:]
    aggregate_truth = np.full((group_count,) + truth_shape, np.nan, dtype=np.float64)
    for group_index in range(group_count):
        selected = inverse == group_index
        group_mask = mask[selected]
        aggregate_mask[group_index] = np.any(group_mask, axis=0)
        group_truth = occurrences["truth"][selected]
        if group_truth.ndim == 1:
            valid = group_mask[:, 0]
            if np.any(valid):
                aggregate_truth[group_index] = group_truth[valid][0]
        else:
            for component in range(group_truth.shape[1]):
                valid = group_mask[:, component]
                if np.any(valid):
                    aggregate_truth[group_index, component] = group_truth[valid, component][0]
    aggregate["truth"] = aggregate_truth
    aggregate["mask"] = aggregate_mask[:, 0] if occurrences["mask"].ndim == 1 else aggregate_mask
    paper_xy = np.full((group_count, 2), np.nan, dtype=np.float64)
    if "paper_xy_cm" in occurrences:
        for group_index in range(group_count):
            selected_values = occurrences["paper_xy_cm"][inverse == group_index]
            finite = np.all(np.isfinite(selected_values), axis=1)
            if np.any(finite):
                paper_xy[group_index] = selected_values[finite][0]
    aggregate["paper_xy_cm"] = paper_xy
    if target_name == "position":
        joint_probabilities = _mean_by_group(
            occurrences["joint_probabilities"], inverse, group_count
        )
        aggregate["joint_probabilities"] = joint_probabilities
        aggregate["x_probabilities"] = joint_probabilities.sum(axis=1)
        aggregate["y_probabilities"] = joint_probabilities.sum(axis=2)
        aggregate["truth_xy_cm"] = paper_xy.copy()
    elif target_name == "area":
        aggregate["probabilities"] = _mean_by_group(
            occurrences["probabilities"], inverse, group_count
        )
    else:
        aggregate["prediction"] = _mean_by_group(occurrences["prediction"], inverse, group_count)
    return aggregate


def evaluate_and_save(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    prepared: PreparedData,
    output_directory: Path | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    occurrences, loss = collect_anchor_predictions(
        model,
        loader,
        device,
        prepared.target,
        prepared.stats,
    )
    session_ids = [session.info.session_id for session in prepared.sessions]
    session_slugs = [session.info.slug for session in prepared.sessions]
    if prepared.target == "position":
        prediction = {"joint_probabilities": occurrences["joint_probabilities"]}
    elif prepared.target == "area":
        prediction = {"probabilities": occurrences["probabilities"]}
    else:
        prediction = {"values": occurrences["prediction"]}
    metrics = summarize_metrics(
        prepared.target,
        occurrences["truth"],
        prediction,
        occurrences["mask"],
        occurrences["session_index"],
        session_slugs,
        paper_xy_cm=occurrences.get("paper_xy_cm"),
    )
    metrics["pooled_validation_loss"] = loss
    aggregate = aggregate_physical_tokens(occurrences, prepared.target)
    aggregate["session_ids"] = np.asarray(session_ids)
    aggregate["session_slugs"] = np.asarray(session_slugs)
    if output_directory is not None:
        np.savez_compressed(output_directory / "aggregated_predictions.npz", **aggregate)
    return metrics, aggregate

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .core import (
    AREA_CLASS_NAMES,
    POSITION_INTERIOR_BINS,
    SOURCE_NAMES,
    position_bin_centers,
)


def position_metrics(
    truth: np.ndarray,
    joint_probabilities: np.ndarray,
    paper_xy_cm: np.ndarray,
) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    joint = np.asarray(joint_probabilities, dtype=np.float64).reshape(
        -1, POSITION_INTERIOR_BINS, POSITION_INTERIOR_BINS
    )
    paper_xy_cm = np.asarray(paper_xy_cm, dtype=np.float64).reshape(-1, 2)
    if not (len(truth) == len(joint) == len(paper_xy_cm)):
        raise ValueError("Position metric inputs have incompatible lengths")
    if len(truth) == 0:
        return {
            "top_4": float("nan"),
            "top_8": float("nan"),
            "top_16": float("nan"),
            "map_euclidean_error_cm": float("nan"),
            "expected_euclidean_error_cm": float("nan"),
            "count": 0,
        }
    flat = joint.reshape(len(joint), -1)
    result: dict[str, float | int] = {"count": int(len(truth))}
    for top_k in (4, 8, 16):
        candidates = np.argpartition(-flat, kth=top_k - 1, axis=1)[:, :top_k]
        result[f"top_{top_k}"] = float(
            np.mean(np.any(candidates == truth[:, None], axis=1))
        )
    centers = position_bin_centers()
    maximum = np.argmax(flat, axis=1)
    map_xy = np.column_stack(
        [centers[maximum % POSITION_INTERIOR_BINS], centers[maximum // POSITION_INTERIOR_BINS]]
    )
    x_marginal = joint.sum(axis=1)
    y_marginal = joint.sum(axis=2)
    expected_xy = np.column_stack([x_marginal @ centers, y_marginal @ centers])
    result["map_euclidean_error_cm"] = float(
        np.mean(np.linalg.norm(map_xy - paper_xy_cm, axis=1))
    )
    result["expected_euclidean_error_cm"] = float(
        np.mean(np.linalg.norm(expected_xy - paper_xy_cm, axis=1))
    )
    return result


def confusion_matrix(
    truth: np.ndarray,
    prediction: np.ndarray,
    class_count: int,
) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    if len(truth) != len(prediction):
        raise ValueError("Truth and prediction lengths differ")
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    if len(truth):
        np.add.at(matrix, (truth, prediction), 1)
    return matrix


def area_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    present_classes_only: bool,
) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1, 3)
    if len(truth) != len(probabilities):
        raise ValueError("Area metric inputs have incompatible lengths")
    prediction = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(truth, prediction, 3)
    class_ids = np.unique(truth) if present_classes_only else np.arange(3)
    f1_values: list[float] = []
    for class_id in class_ids:
        true_positive = matrix[class_id, class_id]
        false_positive = matrix[:, class_id].sum() - true_positive
        false_negative = matrix[class_id, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else float(2 * true_positive / denominator))
    return {
        "accuracy": float(np.mean(prediction == truth)) if len(truth) else float("nan"),
        "macro_f1": float(np.mean(f1_values)) if f1_values else float("nan"),
        "confusion_matrix": matrix.tolist(),
        "class_order": list(AREA_CLASS_NAMES),
        "count": int(len(truth)),
        "present_classes": [AREA_CLASS_NAMES[int(index)] for index in np.unique(truth)],
    }


def scalar_regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if len(truth) != len(prediction):
        raise ValueError("Regression metric inputs have incompatible lengths")
    if not len(truth):
        return {"mae": float("nan"), "rmse": float("nan"), "count": 0}
    error = prediction - truth
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "count": int(len(error)),
    }


def vector_regression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    component_names: Sequence[str],
    include_vector_error: bool,
) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if truth.shape != prediction.shape or truth.shape != mask.shape:
        raise ValueError("Vector regression metric inputs have incompatible shapes")
    result: dict[str, Any] = {}
    for index, name in enumerate(component_names):
        result[name] = scalar_regression_metrics(
            truth[mask[:, index], index],
            prediction[mask[:, index], index],
        )
    if include_vector_error:
        joint_mask = np.all(mask, axis=1)
        result["mean_vector_error"] = (
            float(np.mean(np.linalg.norm(prediction[joint_mask] - truth[joint_mask], axis=1)))
            if np.any(joint_mask)
            else float("nan")
        )
        result["vector_count"] = int(np.sum(joint_mask))
    return result


def metric_tree_mean(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    excluded = {"count", "vector_count", "confusion_matrix", "class_order", "present_classes"}
    result: dict[str, Any] = {}
    keys = set.intersection(*(set(record) for record in records))
    for key in sorted(keys):
        if key in excluded:
            continue
        values = [record[key] for record in records]
        if all(isinstance(value, dict) for value in values):
            result[key] = metric_tree_mean(values)
        elif all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            finite = [float(value) for value in values if np.isfinite(value)]
            result[key] = float(np.mean(finite)) if finite else float("nan")
    return result


def summarize_metrics(
    target_name: str,
    truth: np.ndarray,
    prediction: dict[str, np.ndarray],
    mask: np.ndarray,
    session_indices: np.ndarray,
    session_ids: Sequence[str],
    paper_xy_cm: np.ndarray | None = None,
) -> dict[str, Any]:
    session_indices = np.asarray(session_indices, dtype=np.int64).reshape(-1)
    mask = np.asarray(mask, dtype=bool)

    def calculate(selection: np.ndarray, per_session: bool) -> dict[str, Any]:
        if target_name == "position":
            valid = selection & mask.reshape(-1)
            if paper_xy_cm is None:
                raise ValueError("Position metrics require paper coordinates")
            return position_metrics(
                truth[valid],
                prediction["joint_probabilities"][valid],
                np.asarray(paper_xy_cm)[valid],
            )
        if target_name == "area":
            valid = selection & mask.reshape(-1)
            return area_metrics(
                truth[valid],
                prediction["probabilities"][valid],
                present_classes_only=per_session,
            )
        if target_name == "height":
            valid = selection & mask[:, 0]
            return scalar_regression_metrics(truth[valid, 0], prediction["values"][valid, 0])
        component_names = SOURCE_NAMES if target_name == "distance" else ("vx", "vy")
        selected_mask = mask[selection]
        return vector_regression_metrics(
            truth[selection],
            prediction["values"][selection],
            selected_mask,
            component_names,
            include_vector_error=target_name == "velocity",
        )

    all_selection = np.ones(len(session_indices), dtype=bool)
    per_session = {
        session_id: calculate(session_indices == index, per_session=True)
        for index, session_id in enumerate(session_ids)
    }
    return {
        "pooled": calculate(all_selection, per_session=False),
        "equal_session_macro": metric_tree_mean(list(per_session.values())),
        "per_session": per_session,
    }


def checkpoint_selection_metric(
    target_name: str,
    metrics: dict[str, Any],
) -> tuple[str, float, str]:
    """Return ``(name, value, direction)`` for the checkpoint criterion."""
    if target_name == "position":
        return "anchor_top_8", float(metrics["pooled"]["top_8"]), "maximize"
    if target_name == "area":
        return (
            "anchor_equal_session_macro_f1",
            float(metrics["equal_session_macro"]["macro_f1"]),
            "maximize",
        )
    per_session = list(metrics["per_session"].values())
    if target_name == "distance":
        session_values = []
        for values in per_session:
            present = [
                float(values[source]["mae"])
                for source in SOURCE_NAMES
                if int(values[source]["count"]) > 0
            ]
            if present:
                session_values.append(float(np.mean(present)))
        if not session_values:
            raise ValueError("Distance validation contains no present-source labels")
        return (
            "anchor_equal_session_present_source_mae_cm",
            float(np.mean(session_values)),
            "minimize",
        )
    if target_name == "height":
        values = [float(item["mae"]) for item in per_session if int(item["count"]) > 0]
        if not values:
            raise ValueError("Height validation contains no labels")
        return "anchor_equal_session_mae_cm", float(np.mean(values)), "minimize"
    if target_name == "velocity":
        values = [
            float(item["mean_vector_error"])
            for item in per_session
            if int(item["vector_count"]) > 0
        ]
        if not values:
            raise ValueError("Velocity validation contains no joint labels")
        return (
            "anchor_equal_session_mean_vector_error_cm_s",
            float(np.mean(values)),
            "minimize",
        )
    raise ValueError(f"Unknown target: {target_name}")


def selection_is_better(
    candidate_value: float,
    candidate_loss: float,
    candidate_epoch: int,
    best_value: float | None,
    best_loss: float,
    best_epoch: int,
    direction: str,
) -> bool:
    if best_value is None:
        return True
    if direction == "maximize":
        if candidate_value > best_value:
            return True
        if candidate_value < best_value:
            return False
    elif direction == "minimize":
        if candidate_value < best_value:
            return True
        if candidate_value > best_value:
            return False
    else:
        raise ValueError(f"Unknown selection direction: {direction}")
    if candidate_loss < best_loss:
        return True
    if candidate_loss > best_loss:
        return False
    return candidate_epoch < best_epoch

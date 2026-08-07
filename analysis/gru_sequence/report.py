"""Aggregate the LOSO runs (5 held-out sessions x 5 seeds, per architecture version)
into the comparison report: v1 (hidden=64, no dropout/norm) vs v2 (hidden=161,
+dropout, +LayerNorm, tuned LR) vs the existing TCN.

Confusion matrices use a 5-seed majority vote per physical row, with ties broken by
higher mean probability among the tied classes then lower class index -- the same
convention documented for the existing TCN's area confusion-matrix figures
(analysis/reports/spatial-sequence/leave-one-session-out/report.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from analysis.spatial_sequence.core import AREA_CLASS_NAMES, REPOSITORY_ROOT

from .training import RunConfig, default_run_directory

SESSIONS = (
    "mint-only-horizontal-run01",
    "caret-mint-left-lavender-right-run01",
    "caret-lavender-left-mint-right-run01",
    "lavender-only-horizontal-run01",
    "inverted-caret-lavender-left-mint-right-run01",
)
SEEDS = (0, 1, 2, 3, 4)

# From analysis/reports/spatial-sequence/leave-one-session-out/report.md (existing,
# unmodified bidirectional Temporal CNN; area target). Not re-derived -- quoted as-is
# per an explicit scope decision not to re-run TCN for this comparison. That report
# only has macro-F1 mean +/- SD per session; no confusion matrix numbers, per-class
# P/R/F1, or balanced accuracy are available for the TCN, and the LOSO study was never
# run for the Transformer at all (Transformer has within-session results only, which
# are not comparable to LOSO).
EXISTING_TCN_MACRO_F1 = {
    "caret-lavender-left-mint-right-run01": (0.6368, 0.0155),
    "caret-mint-left-lavender-right-run01": (0.4980, 0.0135),
    "inverted-caret-lavender-left-mint-right-run01": (0.5038, 0.0223),
    "lavender-only-horizontal-run01": (0.5623, 0.0656),
    "mint-only-horizontal-run01": (0.6194, 0.0182),
}
EXISTING_TCN_EQUAL_SESSION_SUMMARY = (0.5641, 0.0099)
EXISTING_TCN_PARAMETER_COUNT = 254_787


def _run_directory(held_out_session: str, seed: int, architecture_version: str) -> Path:
    return default_run_directory(
        RunConfig(
            held_out_session=held_out_session,
            seed=seed,
            architecture_version=architecture_version,
        )
    )


def _load_run(
    held_out_session: str, seed: int, architecture_version: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    directory = _run_directory(held_out_session, seed, architecture_version)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    npz = np.load(directory / "aggregated_predictions.npz", allow_pickle=True)
    arrays = {name: npz[name] for name in npz.files}
    return metrics, arrays


def _majority_vote_predictions(
    held_out_session: str, architecture_version: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (truth, majority_vote_prediction) aligned by physical raw_row, across the 5 seeds.

    Rows with an invalid pose at that timestep (mask=False -- e.g. no qualified AprilTag
    pose) have no area label and are excluded, exactly like the official metrics.json
    numbers already exclude them (summarize_metrics's `valid = selection & mask`).
    """
    per_seed = []
    for seed in SEEDS:
        _, arrays = _load_run(held_out_session, seed, architecture_version)
        valid = arrays["mask"].astype(bool)
        order = np.argsort(arrays["raw_row"][valid])
        per_seed.append(
            {
                "raw_row": arrays["raw_row"][valid][order],
                "truth": arrays["truth"][valid][order].astype(np.int64),
                "probabilities": arrays["probabilities"][valid][order],
            }
        )
    reference_rows = per_seed[0]["raw_row"]
    for entry in per_seed[1:]:
        if not np.array_equal(entry["raw_row"], reference_rows):
            raise ValueError(
                f"Seeds disagree on which physical rows have a valid pose for {held_out_session}"
            )
    truth = per_seed[0]["truth"]
    for entry in per_seed[1:]:
        if not np.array_equal(entry["truth"], truth):
            raise ValueError(f"Seeds disagree on ground truth for {held_out_session}")

    votes = np.column_stack([entry["probabilities"].argmax(axis=1) for entry in per_seed])
    mean_probabilities = np.mean([entry["probabilities"] for entry in per_seed], axis=0)
    predictions = np.empty(len(truth), dtype=np.int64)
    for row_index in range(len(truth)):
        counts = np.bincount(votes[row_index], minlength=3)
        top = np.flatnonzero(counts == counts.max())
        predictions[row_index] = top[np.argmax(mean_probabilities[row_index, top])]
    return truth, predictions


def _confusion_and_class_stats(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    present = sorted(np.unique(truth).tolist())
    per_class: dict[str, Any] = {}
    recalls = []
    for class_id in present:
        true_positive = int(matrix[class_id, class_id])
        false_positive = int(matrix[:, class_id].sum() - true_positive)
        support = int(matrix[class_id, :].sum())
        precision = (
            0.0 if (true_positive + false_positive) == 0 else true_positive / (true_positive + false_positive)
        )
        recall = 0.0 if support == 0 else true_positive / support
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        recalls.append(recall)
        per_class[AREA_CLASS_NAMES[class_id]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    accuracy = float(np.mean(truth == prediction))
    balanced_accuracy = float(np.mean(recalls)) if recalls else float("nan")
    macro_f1 = float(np.mean([per_class[AREA_CLASS_NAMES[c]]["f1"] for c in present])) if present else float("nan")
    return {
        "confusion_matrix": matrix.tolist(),
        "class_order": list(AREA_CLASS_NAMES),
        "present_classes": [AREA_CLASS_NAMES[c] for c in present],
        "per_class": per_class,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1_from_majority_vote": macro_f1,
        "count": int(len(truth)),
    }


def _seed_macro_f1_stats(held_out_session: str, architecture_version: str) -> dict[str, Any]:
    values = []
    accuracies = []
    best_epochs = []
    parameter_count = None
    learning_rate = None
    for seed in SEEDS:
        metrics, _ = _load_run(held_out_session, seed, architecture_version)
        values.append(float(metrics["pooled"]["macro_f1"]))
        accuracies.append(float(metrics["pooled"]["accuracy"]))
        best_epochs.append(int(metrics["best_epoch"]))
        parameter_count = int(metrics["parameter_count"])
    resolved = json.loads(
        (_run_directory(held_out_session, SEEDS[0], architecture_version) / "resolved_config.json").read_text()
    )
    learning_rate = resolved["learning_rate"]
    return {
        "macro_f1_mean": float(np.mean(values)),
        "macro_f1_std": float(np.std(values)),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "median_best_epoch": float(np.median(best_epochs)),
        "parameter_count": parameter_count,
        "learning_rate": learning_rate,
        "per_seed_macro_f1": values,
    }


def format_confusion_matrix(matrix: list[list[int]], class_order: list[str]) -> str:
    header = "| truth \\ predicted | " + " | ".join(class_order) + " |"
    separator = "|---|" + "---:|" * len(class_order)
    rows = [header, separator]
    for row_name, row in zip(class_order, matrix):
        rows.append(f"| **{row_name}** | " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rows)


def _collect(architecture_version: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    per_session_gru: dict[str, dict[str, Any]] = {}
    per_session_confusion: dict[str, dict[str, Any]] = {}
    for session in SESSIONS:
        per_session_gru[session] = _seed_macro_f1_stats(session, architecture_version)
        truth, prediction = _majority_vote_predictions(session, architecture_version)
        per_session_confusion[session] = _confusion_and_class_stats(truth, prediction)
    return per_session_gru, per_session_confusion


def _equal_session_summary(per_session_gru: dict[str, dict[str, Any]]) -> tuple[float, float]:
    means = [stats["macro_f1_mean"] for stats in per_session_gru.values()]
    return float(np.mean(means)), float(np.std(means))


def _breakdown_section(
    title: str, per_session_gru: dict[str, dict[str, Any]], per_session_confusion: dict[str, dict[str, Any]]
) -> list[str]:
    lines = [f"## {title}", ""]
    for session in SESSIONS:
        gru = per_session_gru[session]
        confusion = per_session_confusion[session]
        lines.append(f"### {session}")
        lines.append("")
        lines.append(
            f"5-seed accuracy: {gru['accuracy_mean']:.4f} +/- {gru['accuracy_std']:.4f} | "
            f"5-seed macro-F1: {gru['macro_f1_mean']:.4f} +/- {gru['macro_f1_std']:.4f} | "
            f"per-seed macro-F1: {[round(v, 4) for v in gru['per_seed_macro_f1']]}"
        )
        lines.append("")
        lines.append(
            f"Majority-vote-across-seeds confusion matrix (n={confusion['count']}): "
            f"accuracy {confusion['accuracy']:.4f}, balanced accuracy "
            f"{confusion['balanced_accuracy']:.4f}, macro-F1 "
            f"{confusion['macro_f1_from_majority_vote']:.4f}, present classes: "
            f"{', '.join(confusion['present_classes'])}"
        )
        lines.append("")
        lines.append(format_confusion_matrix(confusion["confusion_matrix"], confusion["class_order"]))
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|---|---:|---:|---:|---:|")
        for class_name in AREA_CLASS_NAMES:
            if class_name in confusion["per_class"]:
                stats = confusion["per_class"][class_name]
                lines.append(
                    f"| {class_name} | {stats['precision']:.4f} | {stats['recall']:.4f} | "
                    f"{stats['f1']:.4f} | {stats['support']} |"
                )
            else:
                lines.append(f"| {class_name} | -- | -- | -- | 0 (not present in this session) |")
        lines.append("")
    return lines


def generate_report(architecture_version: str = "v2") -> str:
    """Full comparison: TCN (quoted) vs. v1 vs. v2 (if v2 runs exist), else TCN vs. v1 only."""
    v1_gru, v1_confusion = _collect("v1")
    v1_param_count = next(iter(v1_gru.values()))["parameter_count"]
    v1_mean, v1_sd = _equal_session_summary(v1_gru)

    have_v2 = all(
        (_run_directory(session, seed, "v2") / "metrics.json").is_file()
        for session in SESSIONS
        for seed in SEEDS
    )
    v2_gru = v2_confusion = None
    if have_v2:
        v2_gru, v2_confusion = _collect("v2")
        v2_param_count = next(iter(v2_gru.values()))["parameter_count"]
        v2_mean, v2_sd = _equal_session_summary(v2_gru)
        v2_lr = next(iter(v2_gru.values()))["learning_rate"]

    lines: list[str] = []
    lines.append("# Bidirectional GRU vs. bidirectional Temporal CNN -- LOSO area, July 30 sessions")
    lines.append("")
    lines.append(
        "> Held-out-session validation, not an independent test: each fold's held-out "
        "session selects and reports its own best checkpoint (same convention as the "
        "existing LOSO study)."
    )
    lines.append("")

    lines.append("## Parameter counts")
    lines.append("")
    lines.append("| Model | Parameters | Learning rate |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Existing bidirectional Temporal CNN (area head) | {EXISTING_TCN_PARAMETER_COUNT:,} | 1e-4 (unchanged) |")
    lines.append(f"| BiGRU v1 (hidden=64, no dropout/norm) | {v1_param_count:,} | 1e-4 (unchanged, isolates one variable) |")
    if have_v2:
        lines.append(
            f"| **BiGRU v2 (hidden=161, +dropout, +LayerNorm)** | **{v2_param_count:,}** | "
            f"**{v2_lr} (chosen via a 4-candidate probe on one fold)** |"
        )
    lines.append("")

    lines.append("## Main comparison: macro-F1 per held-out session (5-seed mean +/- SD)")
    lines.append("")
    header = "| Held-out session | TCN (existing) | BiGRU v1 |"
    separator = "|---|---:|---:|"
    if have_v2:
        header += " BiGRU v2 |"
        separator += "---:|"
    lines.append(header)
    lines.append(separator)
    for session in SESSIONS:
        tcn_mean, tcn_sd = EXISTING_TCN_MACRO_F1[session]
        row = (
            f"| {session} | {tcn_mean:.4f} +/- {tcn_sd:.4f} | "
            f"{v1_gru[session]['macro_f1_mean']:.4f} +/- {v1_gru[session]['macro_f1_std']:.4f} |"
        )
        if have_v2:
            row += f" {v2_gru[session]['macro_f1_mean']:.4f} +/- {v2_gru[session]['macro_f1_std']:.4f} |"
        lines.append(row)
    summary_row = (
        f"| **Equal-session summary** | **{EXISTING_TCN_EQUAL_SESSION_SUMMARY[0]:.4f} +/- "
        f"{EXISTING_TCN_EQUAL_SESSION_SUMMARY[1]:.4f}** | **{v1_mean:.4f} +/- {v1_sd:.4f}** |"
    )
    if have_v2:
        summary_row += f" **{v2_mean:.4f} +/- {v2_sd:.4f}** |"
    lines.append(summary_row)
    lines.append("")
    lines.append(
        "TCN column is quoted unchanged from `analysis/reports/spatial-sequence/"
        "leave-one-session-out/report.md`; not re-run for this comparison (explicit "
        "scope decision). SD for equal-session summaries is the SD of the 5 "
        "per-session means, matching how the existing report's own SD is computed."
    )
    lines.append("")
    if not have_v2:
        lines.append("*(v2 runs not found yet -- run the v2 sweep and regenerate to add that column.)*")
        lines.append("")

    if have_v2:
        lines.append("## Spatial diagnostics (ground truth / prediction / disagreement / class probabilities)")
        lines.append("")
        lines.append(
            "Identical layout/rendering to the existing TCN figures -- generated by calling "
            "`analysis.spatial_sequence.visualization.render_cross_seed_diagnostics` unchanged "
            "(see `analysis/gru_sequence/figures.py`), not a reimplementation. Compare against "
            "`analysis/reports/spatial-sequence/leave-one-session-out/figures/"
            "area_bidirectional_tcnn_<session>.png`."
        )
        lines.append("")
        for session in SESSIONS:
            lines.append(f"![BiGRU v2 area diagnostics: {session}](figures/area_bidirectional_bigru-v2_{session}.png)")
            lines.append("")

    lines.extend(_breakdown_section("v1 full per-session breakdown", v1_gru, v1_confusion))
    if have_v2:
        lines.extend(_breakdown_section("v2 full per-session breakdown", v2_gru, v2_confusion))

    report_text = "\n".join(lines)
    output_path = REPOSITORY_ROOT / "analysis" / "reports" / "gru-sequence"
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "report.md").write_text(report_text + "\n", encoding="utf-8")
    summary_json: dict[str, Any] = {
        "tcn_parameter_count": EXISTING_TCN_PARAMETER_COUNT,
        "v1": {
            "parameter_count": v1_param_count,
            "equal_session_macro_f1_mean": v1_mean,
            "equal_session_macro_f1_sd": v1_sd,
            "per_session": {s: {**v1_gru[s], "majority_vote": v1_confusion[s]} for s in SESSIONS},
        },
    }
    if have_v2:
        summary_json["v2"] = {
            "parameter_count": v2_param_count,
            "learning_rate": v2_lr,
            "equal_session_macro_f1_mean": v2_mean,
            "equal_session_macro_f1_sd": v2_sd,
            "per_session": {s: {**v2_gru[s], "majority_vote": v2_confusion[s]} for s in SESSIONS},
        }
    (output_path / "summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_text

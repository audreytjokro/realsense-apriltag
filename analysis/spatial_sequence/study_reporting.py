from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .core import (
    REPOSITORY_ROOT,
    SOURCE_NAMES,
    discover_sessions,
    load_annotation,
)
from .manifest import MANIFEST_PATH, ManifestRun, SEEDS, validate_run_manifest
from .visualization import (
    render_cross_seed_diagnostics,
    render_cross_seed_parity_scatter,
)


REPORT_ROOT = REPOSITORY_ROOT / "analysis" / "reports" / "spatial-sequence"


def _metrics(run: ManifestRun) -> dict[str, Any]:
    return json.loads((run.run_directory / "metrics.json").read_text(encoding="utf-8"))


def _aggregate(run: ManifestRun) -> dict[str, np.ndarray]:
    with np.load(run.run_directory / "aggregated_predictions.npz") as record:
        return {name: record[name].copy() for name in record.files}


def _primary_axis_label(target: str) -> str:
    return "Present-class macro-F1" if target == "area" else "Present-source MAE (cm)"


def _session_metric(target: str, metrics: dict[str, Any], session: str) -> float:
    values = metrics["per_session"][session]
    if target == "area":
        return float(values["macro_f1"])
    present = [
        float(values[source]["mae"])
        for source in SOURCE_NAMES
        if int(values[source]["count"]) > 0
    ]
    if not present:
        raise ValueError(f"Distance metrics contain no present source for {session}")
    return float(np.mean(present))


def _five_seed_group(runs: Sequence[ManifestRun]) -> list[ManifestRun]:
    ordered = sorted(runs, key=lambda run: run.seed)
    if [run.seed for run in ordered] != list(SEEDS):
        raise ValueError("Every study group must contain seeds 0-4 exactly once")
    return ordered


def _save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.clear()
    return path


def _sequence_groups(
    runs: Sequence[ManifestRun],
) -> dict[tuple[str, int], list[ManifestRun]]:
    groups: dict[tuple[str, int], list[ManifestRun]] = defaultdict(list)
    for run in runs:
        groups[(run.target, run.sequence_length)].append(run)
    return {key: _five_seed_group(value) for key, value in groups.items()}


def _plot_sequence_primary(
    groups: dict[tuple[str, int], list[ManifestRun]], path: Path
) -> Path:
    lengths = (6, 12, 18, 24)
    figure = Figure(figsize=(11.5, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    for panel, target in enumerate(("area", "distance"), 1):
        axis = figure.add_subplot(1, 2, panel)
        matrix = np.asarray(
            [
                [float(_metrics(run)["best_selection_value"]) for run in groups[(target, length)]]
                for length in lengths
            ],
            dtype=float,
        ).T
        for seed, values in enumerate(matrix):
            axis.plot(lengths, values, marker="o", alpha=0.48, linewidth=1.0, label=f"Seed {seed}")
        axis.errorbar(
            lengths,
            matrix.mean(axis=0),
            yerr=matrix.std(axis=0, ddof=0),
            color="black",
            marker="o",
            linewidth=2.2,
            capsize=4,
            label="Mean ± SD",
        )
        axis.set_xticks(lengths)
        axis.set_xlabel("Sequence length (rows)")
        axis.set_ylabel(_primary_axis_label(target))
        axis.set_title(target.title())
        axis.grid(True, alpha=0.25)
        if panel == 2:
            axis.legend(fontsize=8)
    figure.suptitle("Sequence-Length Ablation — Bidirectional TCNN", fontsize=18)
    return _save(figure, path)


def _plot_sequence_sessions(
    groups: dict[tuple[str, int], list[ManifestRun]], path: Path
) -> Path:
    lengths = (6, 12, 18, 24)
    sessions = sorted(_metrics(groups[("area", 24)][0])["per_session"])
    figure = Figure(figsize=(12, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    legend_handles = None
    legend_labels = None
    for panel, target in enumerate(("area", "distance"), 1):
        axis = figure.add_subplot(1, 2, panel)
        for session in sessions:
            values = [
                np.mean(
                    [
                        _session_metric(target, _metrics(run), session)
                        for run in groups[(target, length)]
                    ]
                )
                for length in lengths
            ]
            axis.plot(lengths, values, marker="o", linewidth=1.5, label=session)
        axis.set_xticks(lengths)
        axis.set_xlabel("Sequence length (rows)")
        axis.set_ylabel(_primary_axis_label(target))
        axis.set_title(target.title())
        axis.grid(True, alpha=0.25)
        if panel == 1:
            legend_handles, legend_labels = axis.get_legend_handles_labels()
    figure.legend(
        legend_handles,
        legend_labels,
        loc="outside lower center",
        ncol=2,
        fontsize=8,
    )
    figure.suptitle("Per-Session Sequence-Length Results", fontsize=18)
    return _save(figure, path)


def _plot_sequence_epochs(
    groups: dict[tuple[str, int], list[ManifestRun]], path: Path
) -> Path:
    lengths = (6, 12, 18, 24)
    figure = Figure(figsize=(11.5, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    for panel, target in enumerate(("area", "distance"), 1):
        axis = figure.add_subplot(1, 2, panel)
        matrix = np.asarray(
            [
                [float(_metrics(run)["best_epoch"]) for run in groups[(target, length)]]
                for length in lengths
            ],
            dtype=float,
        ).T
        for seed, values in enumerate(matrix):
            axis.plot(
                lengths,
                values,
                marker="o",
                alpha=0.5,
                linewidth=1.0,
                label=f"Seed {seed}",
            )
        axis.plot(
            lengths,
            np.median(matrix, axis=0),
            color="black",
            marker="o",
            linewidth=2.2,
            label="Median",
        )
        axis.set_xticks(lengths)
        axis.set_xlabel("Sequence length (rows)")
        axis.set_ylabel("Best epoch")
        axis.set_title(target.title())
        axis.grid(True, alpha=0.25)
        if panel == 2:
            axis.legend(fontsize=8)
    figure.suptitle("Checkpoint Selection by Sequence Length", fontsize=18)
    return _save(figure, path)


def generate_sequence_length_report(manifest_path: Path) -> Path:
    runs = validate_run_manifest(manifest_path, require_outputs=True)
    if not runs or runs[0].study != "sequence-length":
        raise ValueError("Expected the sequence-length manifest")
    groups = _sequence_groups(runs)
    directory = REPORT_ROOT / "sequence-length"
    figures = directory / "figures"
    primary = _plot_sequence_primary(groups, figures / "primary_metrics.png")
    sessions = _plot_sequence_sessions(groups, figures / "per_session_metrics.png")
    epochs = _plot_sequence_epochs(groups, figures / "best_epochs.png")
    task_means = {
        (target, length): np.mean(
            [float(_metrics(run)["best_selection_value"]) for run in groups[(target, length)]]
        )
        for target in ("area", "distance")
        for length in (6, 12, 18, 24)
    }
    best_area_length = max((6, 12, 18, 24), key=lambda length: task_means[("area", length)])
    best_distance_length = min(
        (6, 12, 18, 24), key=lambda length: task_means[("distance", length)]
    )
    lines = [
        "# Sequence-Length Ablation",
        "",
        "> Exploratory within-session validation; the same anchors select checkpoints and estimate performance.",
        "",
        "Bidirectional Temporal CNN models use the unchanged kernel-5 dilation-1/2/3 backbone. Lengths 6, 12, 18, and 24 share the canonical physical split and validation anchors; every length enumerates all legal training windows. Normalization gives every training physical row one vote.",
        "",
        f"![Primary metrics](figures/{primary.name})",
        "",
        f"![Per-session metrics](figures/{sessions.name})",
        "",
        f"![Best epochs](figures/{epochs.name})",
        "",
        "| Task | Length | Main metric mean ± SD | Mean minus length 24 | Median best epoch |",
        "|---|---:|---:|---:|---:|",
    ]
    for target in ("area", "distance"):
        reference = np.mean(
            [float(_metrics(run)["best_selection_value"]) for run in groups[(target, 24)]]
        )
        for length in (6, 12, 18, 24):
            values = np.asarray(
                [float(_metrics(run)["best_selection_value"]) for run in groups[(target, length)]]
            )
            best_epochs = np.asarray([int(_metrics(run)["best_epoch"]) for run in groups[(target, length)]])
            lines.append(
                f"| {target} | {length} | {values.mean():.4f} ± {values.std(ddof=0):.4f} | "
                f"{values.mean() - reference:+.4f} | {np.median(best_epochs):.1f} |"
            )
    lines.extend(
        [
            "",
            "## Per-session means",
            "",
            "Each cell averages seeds 0-4. Area is present-class macro-F1; distance is present-source MAE in cm.",
            "",
            "| Task | Session | Length 6 | Length 12 | Length 18 | Length 24 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    session_names = sorted(_metrics(groups[("area", 24)][0])["per_session"])
    for target in ("area", "distance"):
        for session in session_names:
            values = []
            for length in (6, 12, 18, 24):
                values.append(
                    np.mean(
                        [
                            _session_metric(target, _metrics(run), session)
                            for run in groups[(target, length)]
                        ]
                    )
                )
            lines.append(
                f"| {target} | {session} | "
                + " | ".join(f"{value:.4f}" for value in values)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Length {best_area_length} is numerically best for area and length {best_distance_length} is numerically best for distance. Both are only slightly better than length 24, whereas length 6 is clearly worse in the equal-session summary—especially for distance. The practical conclusion is a plateau around 18-24 rows, not evidence that 18 is intrinsically optimal.",
            "",
            "The distance penalty at length 6 is concentrated in the three dual-source layouts; both single-source sessions change much less. Area is less monotonic across sessions: the mint-only session is unusually strong at length 6, while the other layouts generally benefit from more context. This heterogeneity is why the equal-session summary and the per-session panel should be read together.",
            "",
            "Shorter inputs also tend to select later checkpoints: the median best epoch is 81 for length-6 area and 51 for length-6 distance, versus the low twenties around lengths 18-24. Thus the short-context deficit is not simply caused by stopping those runs too early.",
            "",
            "All five seed points are retained; no significance tests are performed. Positive mean-minus-24 differences are favorable for area, while negative differences are favorable for distance.",
            "",
        ]
    )
    report = directory / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _loso_groups(
    runs: Sequence[ManifestRun],
) -> dict[tuple[str, str], list[ManifestRun]]:
    groups: dict[tuple[str, str], list[ManifestRun]] = defaultdict(list)
    for run in runs:
        if run.held_out_session is None:
            raise ValueError("LOSO run is missing held_out_session")
        groups[(run.target, run.held_out_session)].append(run)
    return {key: _five_seed_group(value) for key, value in groups.items()}


def _plot_loso_primary(
    groups: dict[tuple[str, str], list[ManifestRun]], path: Path
) -> Path:
    sessions = sorted({session for _, session in groups})
    figure = Figure(figsize=(14, 5.3), constrained_layout=True)
    FigureCanvasAgg(figure)
    for panel, target in enumerate(("area", "distance"), 1):
        axis = figure.add_subplot(1, 2, panel)
        matrix = np.asarray(
            [
                [float(_metrics(run)["best_selection_value"]) for run in groups[(target, session)]]
                for session in sessions
            ]
        ).T
        x = np.arange(len(sessions))
        for values in matrix:
            axis.plot(x, values, marker="o", alpha=0.45, linewidth=1.0)
        axis.errorbar(
            x,
            matrix.mean(axis=0),
            yerr=matrix.std(axis=0, ddof=0),
            color="black",
            marker="o",
            linewidth=2.2,
            capsize=4,
        )
        axis.set_xticks(x, sessions, rotation=25, ha="right", fontsize=8)
        axis.set_ylabel(_primary_axis_label(target))
        axis.set_title(target.title())
        axis.grid(True, alpha=0.25)
    figure.suptitle("Leave-One-Session-Out Validation — Bidirectional TCNN", fontsize=18)
    return _save(figure, path)


def _plot_loso_area_confusions(
    groups: dict[tuple[str, str], list[ManifestRun]], path: Path
) -> Path:
    sessions = sorted(session for target, session in groups if target == "area")
    figure = Figure(figsize=(17, 4.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    image = None
    for panel, session in enumerate(sessions, 1):
        matrix = np.sum(
            [
                np.asarray(_metrics(run)["per_session"][session]["confusion_matrix"], dtype=float)
                for run in groups[("area", session)]
            ],
            axis=0,
        )
        row_total = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_total, out=np.zeros_like(matrix), where=row_total > 0)
        axis = figure.add_subplot(1, len(sessions), panel)
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{int(matrix[row, column])}\n{normalized[row, column]:.2f}", ha="center", va="center", fontsize=8)
        axis.set_xticks(range(3), ("none", "mint", "lavender"), rotation=30, ha="right")
        axis.set_yticks(range(3), ("none", "mint", "lavender"))
        axis.set_title(session, fontsize=9)
        axis.set_xlabel("Predicted")
        if panel == 1:
            axis.set_ylabel("Ground truth")
    if image is not None:
        figure.colorbar(image, ax=figure.axes, shrink=0.82, label="Row-normalized fraction")
    figure.suptitle("LOSO Area Confusion Matrices", fontsize=18)
    return _save(figure, path)


def generate_loso_report(manifest_path: Path) -> Path:
    runs = validate_run_manifest(manifest_path, require_outputs=True)
    if not runs or runs[0].study != "leave-one-session-out":
        raise ValueError("Expected the leave-one-session-out manifest")
    groups = _loso_groups(runs)
    directory = REPORT_ROOT / "leave-one-session-out"
    figures = directory / "figures"
    primary = _plot_loso_primary(groups, figures / "primary_metrics.png")
    confusion = _plot_loso_area_confusions(groups, figures / "area_confusion_matrices.png")
    session_layouts = {
        info.run_directory.name: load_annotation(info).source_polygons_paper_cm
        for info in discover_sessions()
    }
    area_spatial_paths = []
    for target, session in sorted(groups):
        if target != "area":
            continue
        outputs = render_cross_seed_diagnostics(
            "area",
            [_aggregate(run) for run in groups[(target, session)]],
            figures / f"area_bidirectional_tcnn_{session}.png",
            architecture="temporal-cnn",
            temporal_mode="bidirectional",
            session_layouts=session_layouts,
            constrained_layout=False,
            _session_slug=session,
        )
        if len(outputs) != 1:
            raise AssertionError(
                f"Expected one LOSO area spatial figure for {session}, found {len(outputs)}"
            )
        area_spatial_paths.extend(outputs)
    if len(area_spatial_paths) != 5:
        raise AssertionError(
            f"Expected five LOSO area spatial figures, found {len(area_spatial_paths)}"
        )
    parity_paths = []
    for target, session in sorted(groups):
        if target != "distance":
            continue
        parity_paths.append(
            render_cross_seed_parity_scatter(
                "distance",
                [_aggregate(run) for run in groups[(target, session)]],
                figures / f"distance_parity_{session}.png",
                architecture="temporal-cnn",
                temporal_mode="bidirectional",
            )
        )
    if len(parity_paths) != 5:
        raise AssertionError(f"Expected five LOSO distance parity figures, found {len(parity_paths)}")
    sessions = sorted({session for _, session in groups})
    canonical_runs = validate_run_manifest(MANIFEST_PATH, require_outputs=False)
    canonical_reference: dict[str, float] = {}
    for target in ("area", "distance"):
        values = [
            float(_metrics(run)["best_selection_value"])
            for run in canonical_runs
            if run.target == target
            and run.architecture == "temporal-cnn"
            and run.temporal_mode == "bidirectional"
        ]
        if len(values) != 5:
            raise ValueError(f"Expected five canonical reference runs for {target}")
        canonical_reference[target] = float(np.mean(values))

    summary_values: dict[str, np.ndarray] = {}
    lines = [
        "# Leave-One-Session-Out Validation",
        "",
        "> Cross-session validation, not an independent test: each held-out session selects and reports its fold's best checkpoint.",
        "",
        "Each fold trains on every annotated row from four sessions and validates on every full-context anchor from the fifth. Input and target statistics use each training physical row exactly once. Models are the unchanged length-24 bidirectional Temporal CNN.",
        "",
        f"![Primary metrics](figures/{primary.name})",
        "",
        f"![Area confusion matrices](figures/{confusion.name})",
        "",
        "## Area spatial diagnostics",
        "",
        "Each held-out session is shown separately. Prediction is the per-anchor majority vote across seeds 0-4; class probabilities are arithmetic means and disagreement is the mean pairwise probability difference.",
        "",
    ]
    for path in area_spatial_paths:
        lines.extend([f"![{path.stem}](figures/{path.name})", ""])
    lines.extend(
        [
        "| Task | Held-out session | Main metric mean ± SD | Median best epoch |",
        "|---|---|---:|---:|",
        ]
    )
    for target in ("area", "distance"):
        seed_session = []
        for session in sessions:
            values = np.asarray(
                [float(_metrics(run)["best_selection_value"]) for run in groups[(target, session)]]
            )
            seed_session.append(values)
            epochs = np.asarray(
                [int(_metrics(run)["best_epoch"]) for run in groups[(target, session)]]
            )
            lines.append(
                f"| {target} | {session} | {values.mean():.4f} ± {values.std(ddof=0):.4f} | "
                f"{np.median(epochs):.1f} |"
            )
        session_balanced_by_seed = np.mean(np.stack(seed_session, axis=1), axis=1)
        summary_values[target] = session_balanced_by_seed
        lines.append(
            f"| {target} | **Equal-session summary** | **{session_balanced_by_seed.mean():.4f} ± {session_balanced_by_seed.std(ddof=0):.4f}** | — |"
        )
    area_delta = float(summary_values["area"].mean() - canonical_reference["area"])
    distance_delta = float(
        summary_values["distance"].mean() - canonical_reference["distance"]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The equal-session LOSO area macro-F1 is {summary_values['area'].mean():.4f}, {abs(area_delta):.4f} lower than the canonical within-session length-24 value ({canonical_reference['area']:.4f}). LOSO distance MAE is {summary_values['distance'].mean():.4f} cm, {distance_delta:.4f} cm higher than the within-session value ({canonical_reference['distance']:.4f} cm). These gaps are descriptive rather than paired test effects: LOSO scores an entire held-out session, while the canonical protocol scores centered within-session anchors.",
            "",
            "Generalization varies substantially by session. Area is strongest when holding out caret-lavender-left-mint-right and weakest for the swapped-caret and inverted-caret layouts. The confusion matrices show why: `none` recall stays high, but odor-class recall often collapses toward `none`—especially mint in the inverted-caret fold and both odor classes in the swapped-caret fold. Macro-F1 exposes this failure despite high overall accuracy.",
            "",
            "Distance transfers better than area but remains systematically worse than within-session validation. The parity plots show compressed predictions: near-source distances are often overestimated and large distances underestimated. The swapped-caret session is the hardest distance fold; the two single-source sessions are easier, although they exercise only one output head.",
            "",
            "Because the held-out session selects its own best epoch, this is cross-session validation—not an independent test estimate. A future deployment estimate needs another untouched session or a nested selection protocol.",
            "",
            "Distance parity figures retain all five seed predictions and omit absent-source heads in single-source sessions. No significance tests are performed.",
            "",
        ]
    )
    report = directory / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def generate_study_report(manifest_path: Path) -> Path:
    runs = validate_run_manifest(manifest_path)
    if not runs:
        raise ValueError("Manifest is empty")
    if runs[0].study == "sequence-length":
        return generate_sequence_length_report(manifest_path)
    if runs[0].study == "leave-one-session-out":
        return generate_loso_report(manifest_path)
    raise ValueError("Canonical reports are generated by reporting.generate_report")

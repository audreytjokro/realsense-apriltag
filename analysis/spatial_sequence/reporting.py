from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .core import REPOSITORY_ROOT, build_prepared_data, discover_sessions, load_annotation
from .manifest import (
    MANIFEST_PATH,
    ManifestRun,
    representative_position_runs,
    validate_run_manifest,
)
from .training import evaluate_checkpoint
from .visualization import (
    render_cross_seed_diagnostics,
    render_cross_seed_parity_scatter,
    render_run_diagnostics,
)


REPORT_DIRECTORY = REPOSITORY_ROOT / "analysis" / "reports" / "spatial-sequence"
REPORT_PATH = (
    REPORT_DIRECTORY / "report.md"
)
FIGURE_DIRECTORY = REPORT_DIRECTORY / "figures"


def _read_metrics(run: ManifestRun) -> dict[str, Any]:
    with (run.run_directory / "metrics.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_history(run: ManifestRun) -> pd.DataFrame:
    return pd.read_csv(run.run_directory / "history.csv")


def _logical_key(run: ManifestRun) -> tuple[str, str, str, str]:
    return (
        run.target,
        str(run.session or "pooled"),
        run.architecture,
        run.temporal_mode,
    )


def _groups(runs: Sequence[ManifestRun]) -> dict[tuple[str, str, str, str], list[ManifestRun]]:
    grouped: dict[tuple[str, str, str, str], list[ManifestRun]] = defaultdict(list)
    for run in runs:
        grouped[_logical_key(run)].append(run)
    for key, values in grouped.items():
        values.sort(key=lambda run: run.seed)
        if [run.seed for run in values] != list(range(5)):
            raise ValueError(f"Logical configuration lacks seeds 0-4: {key}")
    if len(grouped) != 36:
        raise ValueError(f"Expected 36 logical configurations, found {len(grouped)}")
    return dict(grouped)


def classify_seed_convergence(run: ManifestRun) -> str:
    history = _read_history(run)
    metrics = _read_metrics(run)
    best_epoch = int(metrics["best_epoch"])
    direction = str(history["selection_direction"].iloc[0])
    values = history["selection_metric_value"].to_numpy(dtype=float)
    train = history["train_loss"].to_numpy(dtype=float)
    best_value = float(values[best_epoch - 1])
    tail_value = float(np.mean(values[-10:]))
    train_continued_down = float(np.mean(train[-10:])) <= float(train[best_epoch - 1]) * 0.99
    if direction == "maximize":
        deteriorated = best_value - tail_value >= 0.03
    else:
        deteriorated = (tail_value - best_value) / max(abs(best_value), 1e-12) >= 0.05
    if best_epoch <= 20 and deteriorated and train_continued_down:
        return "fast-overfit"

    tail = values[-20:]
    slope = float(np.polyfit(np.arange(len(tail), dtype=float), tail, 1)[0])
    projected_change = slope * max(len(tail) - 1, 1)
    if direction == "maximize":
        trending_better = projected_change >= 0.01
    else:
        trending_better = -projected_change / max(abs(float(np.mean(tail))), 1e-12) >= 0.01
    if best_epoch >= 90 or trending_better:
        return "still-improving"
    return "plateau"


def summarize_convergence(
    runs: Sequence[ManifestRun],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for key, group in _groups(runs).items():
        labels = {run.seed: classify_seed_convergence(run) for run in group}
        counts = {name: list(labels.values()).count(name) for name in set(labels.values())}
        if counts.get("fast-overfit", 0) >= 3:
            consensus = "fast-overfit"
        elif counts.get("still-improving", 0) >= 3:
            consensus = "still-improving"
        else:
            consensus = "mixed/plateau"
        result[key] = {"consensus": consensus, "seed_labels": labels}
    return result


def _plot_loss_curves(groups: dict[tuple[str, str, str, str], list[ManifestRun]], output: Path) -> None:
    figure, axes = plt.subplots(6, 6, figsize=(20, 17), sharex=True)
    for axis, (key, runs) in zip(axes.flat, sorted(groups.items())):
        for run in runs:
            history = _read_history(run)
            axis.plot(
                history["epoch"],
                history["train_loss"],
                color="tab:blue",
                alpha=0.25,
                linewidth=0.8,
            )
            axis.plot(
                history["epoch"],
                history["dense_validation_loss"],
                color="tab:orange",
                alpha=0.25,
                linewidth=0.8,
            )
            axis.plot(
                history["epoch"],
                history["anchor_validation_loss"],
                color="tab:red",
                alpha=0.35,
                linewidth=0.8,
            )
        target, session, architecture, mode = key
        model = "TCNN" if architecture == "temporal-cnn" else "Transformer"
        axis.set_title(
            f"{target} | {session}\n{mode.title()} {model}",
            fontsize=6,
        )
        axis.grid(alpha=0.15)
    for axis in axes[-1]:
        axis.set_xlabel("epoch", fontsize=7)
    figure.suptitle("Loss Curves", fontsize=18, fontweight="semibold")
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_selection_curves(
    groups: dict[tuple[str, str, str, str], list[ManifestRun]], output: Path
) -> None:
    figure, axes = plt.subplots(6, 6, figsize=(20, 17), sharex=True)
    for axis, (key, runs) in zip(axes.flat, sorted(groups.items())):
        for run in runs:
            history = _read_history(run)
            axis.plot(
                history["epoch"],
                history["selection_metric_value"],
                linewidth=0.8,
                alpha=0.75,
                label=f"seed {run.seed}",
            )
        target, session, architecture, mode = key
        model = "TCNN" if architecture == "temporal-cnn" else "Transformer"
        axis.set_title(
            f"{target} | {session}\n{mode.title()} {model}",
            fontsize=6,
        )
        axis.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=5)
    figure.suptitle("Checkpoint Selection", fontsize=18, fontweight="semibold")
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_seed_comparisons(
    groups: dict[tuple[str, str, str, str], list[ManifestRun]], output: Path
) -> None:
    targets = ("position", "distance", "area", "height", "velocity")
    figure, axes = plt.subplots(1, 5, figsize=(22, 5.5))
    for axis, target in zip(axes, targets):
        selected = [(key, runs) for key, runs in sorted(groups.items()) if key[0] == target]
        labels = []
        for x, (key, runs) in enumerate(selected):
            values = np.asarray(
                [float(_read_metrics(run)["best_selection_value"]) for run in runs]
            )
            axis.scatter(np.full(5, x), values, s=17, alpha=0.75, color="tab:blue")
            axis.errorbar(
                x,
                float(np.mean(values)),
                yerr=float(np.std(values, ddof=0)),
                marker="D",
                color="black",
                capsize=3,
            )
            _, session, architecture, mode = key
            model = "tcnn" if architecture == "temporal-cnn" else "trans"
            labels.append(f"{session}\n{mode[:3]}-{model}")
        axis.set_xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=6)
        axis.set_title(target)
        axis.set_ylabel(
            "Top-8 / F1 (higher is better)"
            if target in {"position", "area"}
            else "physical error (lower is better)"
        )
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Best-Checkpoint Metrics", fontsize=18, fontweight="semibold"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_area_confusions(
    groups: dict[tuple[str, str, str, str], list[ManifestRun]], output: Path
) -> None:
    selected = sorted(
        (key, runs) for key, runs in groups.items() if key[0] == "area"
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 8.8),
        constrained_layout=True,
    )
    class_names = ("none", "mint", "lavender")
    architecture_names = {"temporal-cnn": "TCNN", "transformer": "Transformer"}
    for panel_index, (axis, (key, runs)) in enumerate(zip(axes.flat, selected)):
        panel_row, panel_column = divmod(panel_index, 2)
        matrix = sum(
            np.asarray(_read_metrics(run)["pooled"]["confusion_matrix"], dtype=float)
            for run in runs
        )
        totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        for row in range(3):
            for column in range(3):
                fraction = float(normalized[row, column])
                axis.text(
                    column,
                    row,
                    f"{int(matrix[row, column]):,}\n{fraction:.1%}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if fraction >= 0.55 else "#111827",
                    linespacing=1.25,
                )
        _, _, architecture, mode = key
        axis.set_title(
            f"{mode.title()} {architecture_names[architecture]}",
            fontsize=12,
            fontweight="semibold",
            pad=10,
        )
        axis.set_xticks(range(3), class_names)
        axis.set_yticks(range(3), class_names)
        axis.tick_params(axis="both", labelsize=10, length=0)
        if panel_column == 1:
            axis.tick_params(labelleft=False)
        if panel_row == 1:
            axis.set_xlabel("Predicted class", fontsize=11, labelpad=8)
        if panel_column == 0:
            axis.set_ylabel("True class", fontsize=11, labelpad=8)
        axis.set_xticks(np.arange(-0.5, 3.0, 1.0), minor=True)
        axis.set_yticks(np.arange(-0.5, 3.0, 1.0), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.5)
        axis.tick_params(which="minor", bottom=False, left=False)
    colorbar = figure.colorbar(
        image,
        ax=axes,
        location="right",
        shrink=0.88,
        pad=0.025,
    )
    colorbar.set_label("Fraction within each true class", fontsize=11, labelpad=10)
    colorbar.ax.tick_params(labelsize=9)
    figure.suptitle(
        "Area Confusion Matrices",
        fontsize=18,
        fontweight="semibold",
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _load_aggregate(run: ManifestRun) -> dict[str, np.ndarray]:
    with np.load(run.run_directory / "aggregated_predictions.npz") as archive:
        return {name: archive[name] for name in archive.files}


def _median_performance_seed_index(runs: Sequence[ManifestRun]) -> int:
    values = np.asarray(
        [float(_read_metrics(run)["best_selection_value"]) for run in runs],
        dtype=np.float64,
    )
    median = float(np.median(values))
    return min(
        range(len(runs)),
        key=lambda index: (abs(float(values[index]) - median), runs[index].seed),
    )


def _short_architecture_slug(architecture: str) -> str:
    return "tcnn" if architecture == "temporal-cnn" else architecture


def _spatial_diagnostic_figure_name(
    target: str,
    session: str,
    architecture: str,
    mode: str,
) -> str:
    parts = [target]
    parts.append(mode)
    parts.append(_short_architecture_slug(architecture))
    if session != "pooled":
        parts.append(session)
    return "_".join(parts) + ".png"


def _render_cross_seed_figures(
    groups: dict[tuple[str, str, str, str], list[ManifestRun]],
) -> list[Path]:
    outputs = []
    session_layouts = {
        info.run_directory.name: load_annotation(info).source_polygons_paper_cm
        for info in discover_sessions()
    }
    for key, runs in sorted(groups.items()):
        target, session, architecture, mode = key
        if target == "position":
            continue
        output = FIGURE_DIRECTORY / _spatial_diagnostic_figure_name(
            target,
            session,
            architecture,
            mode,
        )
        outputs.extend(
            render_cross_seed_diagnostics(
                target,
                [_load_aggregate(run) for run in runs],
                output,
                architecture=architecture,
                temporal_mode=mode,
                session_layouts=session_layouts,
            )
        )
    if len(outputs) != 48:
        raise AssertionError(f"Expected 48 cross-seed figures, found {len(outputs)}")
    return outputs


def _render_cross_seed_parity_scatter_figures(
    groups: dict[tuple[str, str, str, str], list[ManifestRun]],
) -> list[Path]:
    outputs: list[Path] = []
    for key, runs in sorted(groups.items()):
        target, session, architecture, mode = key
        if target not in {"distance", "height", "velocity"}:
            continue
        architecture_slug = _short_architecture_slug(architecture)
        outputs.append(
            render_cross_seed_parity_scatter(
                target,
                [_load_aggregate(run) for run in runs],
                FIGURE_DIRECTORY
                / f"scatter_{target}_{mode}_{architecture_slug}.png",
                architecture=architecture,
                temporal_mode=mode,
            )
        )
    if len(outputs) != 12:
        raise AssertionError(
            f"Expected 12 continuous-task parity scatter figures, found {len(outputs)}"
        )
    return outputs


def regenerate_cross_seed_parity_scatter_diagnostics(
    manifest_path: Path = MANIFEST_PATH,
) -> list[Path]:
    """Render the 12 continuous-task parity scatters from existing predictions."""
    runs = validate_run_manifest(manifest_path, require_outputs=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return _render_cross_seed_parity_scatter_figures(_groups(runs))


def regenerate_session_diagnostics(
    target: str,
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[list[Path], list[Path]]:
    """Redraw one pooled target from existing anchor predictions."""
    if target not in {"area", "distance", "height", "velocity"}:
        raise ValueError("Diagnostic redraw requires a pooled target")
    runs = validate_run_manifest(manifest_path, require_outputs=False)
    prepared = build_prepared_data(target)
    per_run_outputs: list[Path] = []
    for run in runs:
        if run.target != target:
            continue
        per_run_outputs.extend(
            render_run_diagnostics(
                prepared,
                _load_aggregate(run),
                run.run_directory / "diagnostics.png",
                architecture=run.architecture,
                temporal_mode=run.temporal_mode,
                seed=run.seed,
            )
        )
    expected_per_run = 100 if target in {"area", "distance"} else 20
    if len(per_run_outputs) != expected_per_run:
        raise AssertionError(
            f"Expected {expected_per_run} per-run {target} figures, "
            f"found {len(per_run_outputs)}"
        )

    session_layouts = {
        info.run_directory.name: load_annotation(
            info, verify_hashes=False
        ).source_polygons_paper_cm
        for info in discover_sessions()
    }
    cross_seed_outputs: list[Path] = []
    for key, group in sorted(_groups(runs).items()):
        key_target, session, architecture, mode = key
        if key_target != target:
            continue
        cross_seed_outputs.extend(
            render_cross_seed_diagnostics(
                key_target,
                [_load_aggregate(run) for run in group],
                FIGURE_DIRECTORY
                / _spatial_diagnostic_figure_name(
                    key_target,
                    session,
                    architecture,
                    mode,
                ),
                architecture=architecture,
                temporal_mode=mode,
                session_layouts=session_layouts,
            )
        )
    expected_cross_seed = 20 if target in {"area", "distance"} else 4
    if len(cross_seed_outputs) != expected_cross_seed:
        raise AssertionError(
            f"Expected {expected_cross_seed} cross-seed {target} figures, "
            f"found {len(cross_seed_outputs)}"
        )
    return per_run_outputs, cross_seed_outputs


def regenerate_area_diagnostics(
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[list[Path], list[Path]]:
    return regenerate_session_diagnostics("area", manifest_path)


def regenerate_all_run_diagnostics(
    manifest_path: Path = MANIFEST_PATH,
) -> list[Path]:
    """Redraw all 340 per-seed diagnostics from saved anchor predictions."""
    runs = validate_run_manifest(manifest_path, require_outputs=False)
    prepared_cache: dict[tuple[str, str | None], Any] = {}
    outputs: list[Path] = []
    for run_index, run in enumerate(runs, 1):
        key = (run.target, run.session)
        if key not in prepared_cache:
            prepared_cache[key] = build_prepared_data(
                run.target,
                session=run.session,
            )
        outputs.extend(
            render_run_diagnostics(
                prepared_cache[key],
                _load_aggregate(run),
                run.run_directory / "diagnostics.png",
                architecture=run.architecture,
                temporal_mode=run.temporal_mode,
                seed=run.seed,
            )
        )
        if run_index % 20 == 0 or run_index == len(runs):
            print(
                f"Run diagnostics: {run_index}/{len(runs)} runs, "
                f"{len(outputs)}/340 figures",
                flush=True,
            )
    if len(outputs) != 340:
        raise AssertionError(f"Expected 340 per-run diagnostics, found {len(outputs)}")
    return outputs


def render_representative_position_videos(runs: Sequence[ManifestRun]) -> list[Path]:
    outputs = []
    for run in representative_position_runs(list(runs)):
        evaluate_checkpoint(
            run.checkpoint_path,
            output_directory=run.run_directory,
            render_video=True,
        )
        output = run.video_path
        if not output.is_file():
            raise FileNotFoundError(output)
        outputs.append(output)
    if len(outputs) != 20:
        raise AssertionError(f"Expected 20 position videos, found {len(outputs)}")
    return outputs


def generate_report(
    manifest_path: Path = MANIFEST_PATH,
    report_path: Path = REPORT_PATH,
    render_position_videos: bool = True,
) -> Path:
    runs = validate_run_manifest(manifest_path, require_outputs=True)
    groups = _groups(runs)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figures = {
        "loss": FIGURE_DIRECTORY / "loss_curves.png",
        "selection": FIGURE_DIRECTORY / "selection_curves.png",
        "comparison": FIGURE_DIRECTORY / "task_seed_comparisons.png",
        "confusion": FIGURE_DIRECTORY / "area_confusion_matrices.png",
    }
    _plot_loss_curves(groups, figures["loss"])
    _plot_selection_curves(groups, figures["selection"])
    _plot_seed_comparisons(groups, figures["comparison"])
    _plot_area_confusions(groups, figures["confusion"])
    cross_seed = _render_cross_seed_figures(groups)
    parity_scatter = _render_cross_seed_parity_scatter_figures(groups)
    expected_figure_count = 4 + len(cross_seed) + len(parity_scatter)
    actual_figure_count = len(list(FIGURE_DIRECTORY.glob("*.png")))
    if expected_figure_count != 64 or actual_figure_count != expected_figure_count:
        raise AssertionError(
            "Report figure directory must contain exactly 64 PNGs; "
            f"found {actual_figure_count}"
        )
    convergence = summarize_convergence(runs)
    convergence_path = report_path.with_name("convergence.json")
    convergence_path.write_text(
        json.dumps(
            {" | ".join(key): value for key, value in convergence.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if render_position_videos:
        videos = render_representative_position_videos(runs)
    else:
        videos = [run.video_path for run in representative_position_runs(runs)]
        absent_videos = [str(path) for path in videos if not path.is_file()]
        if absent_videos:
            raise FileNotFoundError(
                "Representative position videos are missing: "
                + ", ".join(absent_videos)
            )

    relative_figure_root = "figures"
    lines = [
        "# Spatial Sequence: Joint Position and Anchor Evaluation",
        "",
        "> Exploratory validation only. Checkpoints were selected on the same anchor validation rows reported below. There is no independent test set and no deployment-performance claim.",
        "",
        "## Protocol",
        "",
        "The pipeline uses a centered Train-Guard-Validation-Guard-Train split, dense 24-token training, and one formal prediction per shared physical anchor row. Causal models score token 23 from t-23...t; bidirectional models score token 12 from t-12...t+11. Every physical training row contributes once to normalization, and validation rows do not contribute. Position is a paper-only 13x13 joint classification target; paper-exterior position labels are masked.",
        "",
        "The table reports five-seed mean +/- population SD. Every point from seeds 0-4 is retained in the figures; no significance tests were performed.",
        "",
        f"![Loss curves]({relative_figure_root}/{figures['loss'].name})",
        "",
        f"![Selection curves]({relative_figure_root}/{figures['selection'].name})",
        "",
        f"![Seed comparisons]({relative_figure_root}/{figures['comparison'].name})",
        "",
        f"![Area confusion matrices]({relative_figure_root}/{figures['confusion'].name})",
        "",
        "## Five-seed summaries",
        "",
        "| Target | Session | Architecture | Mode | Selection metric mean +/- SD | Convergence |",
        "|---|---|---|---|---:|---|",
    ]
    for key, group in sorted(groups.items()):
        target, session, architecture, mode = key
        values = np.asarray(
            [float(_read_metrics(run)["best_selection_value"]) for run in group]
        )
        metric_name = str(_read_metrics(group[0])["selection_metric_name"])
        lines.append(
            f"| {target} | {session} | {architecture} | {mode} | "
            f"{metric_name}: {np.mean(values):.4f} +/- {np.std(values, ddof=0):.4f} | "
            f"{convergence[key]['consensus']} |"
        )
    lines.extend(
        [
            "",
            "## Main findings",
            "",
            "Across all five targets, the bidirectional Temporal CNN has the best mean task-specific selection metric. Its advantage over its causal counterpart is large for position, distance, and area, modest for velocity, and small for height. This architecture-by-mode interaction suggests that access to the recovery side of the smell sequence is more important than backbone family alone; bidirectional results are offline smoothing results and should not be interpreted as online localization performance.",
            "",
            "| Architecture | Mode | Position session-mean Top-8 | Distance session-equal MAE (cm) | Area session-equal macro-F1 | Height session-equal MAE (cm) | Velocity session-equal vector error (cm/s) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for architecture in ("transformer", "temporal-cnn"):
        for mode in ("causal", "bidirectional"):
            position_groups = [
                group
                for key, group in groups.items()
                if key[0] == "position" and key[2:] == (architecture, mode)
            ]
            position_by_seed = [
                float(
                    np.mean(
                        [
                            _read_metrics(group[seed])["best_selection_value"]
                            for group in position_groups
                        ]
                    )
                )
                for seed in range(5)
            ]

            def pooled_primary(target: str) -> list[float]:
                group = next(
                    values
                    for key, values in groups.items()
                    if key == (target, "pooled", architecture, mode)
                )
                return [float(_read_metrics(run)["best_selection_value"]) for run in group]

            distance = pooled_primary("distance")
            area = pooled_primary("area")
            height = pooled_primary("height")
            velocity = pooled_primary("velocity")
            lines.append(
                f"| {architecture} | {mode} | {np.mean(position_by_seed):.4f} +/- {np.std(position_by_seed, ddof=0):.4f} | "
                f"{np.mean(distance):.4f} +/- {np.std(distance, ddof=0):.4f} | "
                f"{np.mean(area):.4f} +/- {np.std(area, ddof=0):.4f} | "
                f"{np.mean(height):.4f} +/- {np.std(height, ddof=0):.4f} | "
                f"{np.mean(velocity):.4f} +/- {np.std(velocity, ddof=0):.4f} |"
            )

    best_epochs = [int(_read_metrics(run)["best_epoch"]) for run in runs]
    seed_labels = [
        label
        for value in convergence.values()
        for label in value["seed_labels"].values()
    ]
    fast_groups = [
        " / ".join(key)
        for key, value in sorted(convergence.items())
        if value["consensus"] == "fast-overfit"
    ]
    still_improving_groups = [
        key for key, value in convergence.items() if value["consensus"] == "still-improving"
    ]
    lines.extend(
        [
            "",
            "Position remains difficult: even the best overall combination produces a session-equal Top-8 around 0.15, while expected-coordinate errors remain near 8 cm. Expected-coordinate errors vary much less than Top-k or MAP errors, consistent with broad distributions whose probability centroids contract toward the paper center. The direct joint head removes the old factorization artifact, but it does not by itself make the distributions sharp or spatially calibrated.",
            "",
            "Area accuracy is high for every model because `none` is common; macro-F1 is therefore the more informative measure. Distance benefits strongly from future context. Height differences are small and RMSE is much larger than MAE, suggesting a mostly easy target with a smaller number of large errors. Velocity improves only modestly and remains the weakest pooled regression target in physical units.",
            "",
            "## Convergence and overfitting analysis",
            "",
            f"Across 180 seeds, the best epoch ranges from {min(best_epochs)} to {max(best_epochs)} with median {np.median(best_epochs):.1f}; {sum(epoch <= 20 for epoch in best_epochs)} runs select epoch 20 or earlier and {sum(epoch >= 90 for epoch in best_epochs)} select epoch 90 or later. Seed-level labels are: {seed_labels.count('fast-overfit')} fast-overfit, {seed_labels.count('still-improving')} still-improving, and {seed_labels.count('plateau')} plateau.",
            "",
            f"{len(fast_groups)} logical configurations meet the at-least-three-of-five fast-overfit rule: {'; '.join(fast_groups)}.",
            "",
            (
                "No logical configuration is consistently still improving at 100 epochs. "
                if not still_improving_groups
                else f"Consistently still-improving groups: {still_improving_groups}. "
            )
            + "Thus 100 epochs is generally long enough for this dataset; the larger issue is early overfit or a long plateau, not systematic under-training. Best-checkpoint selection prevents the late states from being reported, but it does not create an independent estimate because the same anchors select and evaluate the checkpoint.",
            "",
            "## Recommended next experiments",
            "",
            "1. Add genuinely held-out sessions or trajectories before tuning further. The current validation set is both the checkpoint selector and the reported set.",
            "2. For position, compare spatially structured targets: Gaussian soft labels, distance-aware/optimal-transport loss, or a continuous 2D density head. One-hot CE treats adjacent and opposite bins as equally wrong.",
            "3. Test longer and multi-scale smell-only context. The strong bidirectional advantage is consistent with sensor response/recovery lag; causal deployment may require longer history or an explicit smell-derived dynamics representation.",
            "4. For the fast-overfit groups, test shorter schedules, stronger weight decay/dropout, and smaller backbones. Early stopping would save computation, but its stopping data must remain separate from the final test data.",
            "5. For area, use class-balanced sampling or loss only as a controlled ablation and continue selecting with present-class session macro-F1. Report calibration and per-class recall alongside F1.",
            "6. Add simple baselines in physical units (session-wise constant, persistence when permitted, and smell-only linear/MLP models) before interpreting the small height and velocity differences as meaningful.",
        ]
    )
    lines.extend(
        [
            "",
            "## Position and pooled-task diagnostics",
            "",
            f"Twenty representative position videos were generated ({len(videos)} present in this report run). For each logical position configuration, the representative seed is the one whose Top-8 is nearest the five-seed median; ties prefer the smaller seed. Videos show the direct 13x13 joint heatmap and explicitly mark raw-video alignment as approximate.",
            "",
            f"The {len(cross_seed)} pooled-task cross-seed figures use the same physical anchors across seeds. Area and distance diagnostics are split into five session-specific figures per architecture/mode so that each annotated mint/lavender layout is shown independently; height and velocity retain one figure per architecture/mode. Area figures show GT, per-anchor five-seed majority-vote predictions, mean pairwise seed-probability disagreement, and arithmetic-mean class probabilities. A 2-2-1 vote tie is resolved by the higher mean probability among the tied classes, then by the lower class index. Distance figures show cross-seed median predictions, errors, and seed SD. Height uses a shared GT 1st–99th percentile logarithmic physical-height scale for GT and prediction, with out-of-range values saturated, plus a symlog error scale; this prevents the long upper tail from compressing the main range. Velocity uses all anchors for speed color and at most 120 fixed-length arrows for direction. Independent sigma=1 cm glyphs are used for continuous values; no spatial interpolation or sample merging is used.",
            "",
            f"The {len(parity_scatter)} additional continuous-task parity scatters retain all five seed predictions as identically styled translucent points. Distance uses separate mint and lavender panels, height uses full-range log-log axes, and velocity compares speed magnitudes. Every panel uses matched ground-truth/prediction limits and only a dashed identity reference; the existing spatial diagnostics remain available for location-conditioned errors and velocity direction.",
            "",
            "## Convergence interpretation",
            "",
            "A logical configuration is labeled fast-overfit or still-improving only when at least three of five seeds meet the preregistered rule; all other groups are mixed/plateau. The raw loss and selection curves above remain the primary evidence. Seed-level classifications are preserved in `convergence.json`.",
            "",
            "## Limits",
            "",
            "- Validation anchors selected checkpoints and estimated performance; they are not an independent test set.",
            "- Position models are session-specific. Pooled-task session-equal criteria prevent the longest session from controlling checkpoint selection.",
            "- Raw-frame video alignment is explicitly approximate and is not used for any numerical metric.",
            "- Gaussian glyph diagnostics are renderings of discrete observations, not inferred spatial fields.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

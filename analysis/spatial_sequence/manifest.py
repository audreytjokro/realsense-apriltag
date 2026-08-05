from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .core import REPOSITORY_ROOT


MANIFEST_PATH = Path(__file__).with_name("run_manifest.csv")
SEQUENCE_LENGTH_MANIFEST_PATH = Path(__file__).with_name("sequence_length_manifest.csv")
LOSO_MANIFEST_PATH = Path(__file__).with_name("leave_one_session_out_manifest.csv")
OUTPUT_ROOT = REPOSITORY_ROOT / "output" / "spatial-sequence"
VIDEO_DIRECTORY = OUTPUT_ROOT / "videos"
ARCHITECTURES = ("transformer", "temporal-cnn")
TEMPORAL_MODES = ("causal", "bidirectional")
POOLED_TARGETS = ("distance", "area", "height", "velocity")
POSITION_SESSION_SLUGS = (
    "caret-lavender-left-mint-right-run01",
    "caret-mint-left-lavender-right-run01",
    "inverted-caret-lavender-left-mint-right-run01",
    "lavender-only-horizontal-run01",
    "mint-only-horizontal-run01",
)
SEEDS = tuple(range(5))


def model_slug(architecture: str) -> str:
    return "tcnn" if architecture == "temporal-cnn" else "transformer"


@dataclass(frozen=True)
class ManifestRun:
    run_id: str
    architecture: str
    temporal_mode: str
    target: str
    session: str | None
    seed: int
    run_directory: Path
    study: str = "canonical"
    sequence_length: int = 24
    evaluation_scheme: str = "within-session"
    held_out_session: str | None = None
    reuse_canonical: bool = False

    @property
    def checkpoint_path(self) -> Path:
        return self.run_directory / "best.pt"

    @property
    def video_path(self) -> Path:
        if self.target != "position" or self.session is None:
            raise ValueError("Only position runs have representative videos")
        return VIDEO_DIRECTORY / (
            f"position_{self.session}_{self.temporal_mode}_"
            f"{model_slug(self.architecture)}_seed{self.seed}.mp4"
        )


def expected_configurations() -> set[tuple[str, str, str, str | None, int]]:
    configurations: set[tuple[str, str, str, str | None, int]] = set()
    for architecture in ARCHITECTURES:
        for temporal_mode in TEMPORAL_MODES:
            for seed in SEEDS:
                for session in POSITION_SESSION_SLUGS:
                    configurations.add(
                        (architecture, temporal_mode, "position", session, seed)
                    )
                for target in POOLED_TARGETS:
                    configurations.add((architecture, temporal_mode, target, None, seed))
    return configurations


def expected_sequence_length_configurations() -> set[tuple[str, str, str, int, int]]:
    return {
        ("temporal-cnn", "bidirectional", target, length, seed)
        for target in ("distance", "area")
        for length in (6, 12, 18, 24)
        for seed in SEEDS
    }


def expected_loso_configurations() -> set[tuple[str, str, str, str, int]]:
    return {
        ("temporal-cnn", "bidirectional", target, held_out, seed)
        for target in ("distance", "area")
        for held_out in POSITION_SESSION_SLUGS
        for seed in SEEDS
    }


def load_run_manifest(path: Path = MANIFEST_PATH) -> list[ManifestRun]:
    runs: list[ManifestRun] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        base_columns = {
            "run_id",
            "architecture",
            "temporal_mode",
            "target",
            "session",
            "seed",
            "run_directory",
        }
        extended_columns = base_columns | {
            "study",
            "sequence_length",
            "evaluation_scheme",
            "held_out_session",
            "reuse_canonical",
        }
        columns = frozenset(reader.fieldnames or ())
        if columns not in {frozenset(base_columns), frozenset(extended_columns)}:
            raise ValueError(
                f"Run manifest columns must be {sorted(base_columns)} or "
                f"{sorted(extended_columns)}, "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            relative = Path(row["run_directory"])
            runs.append(
                ManifestRun(
                    run_id=row["run_id"],
                    architecture=row["architecture"],
                    temporal_mode=row["temporal_mode"],
                    target=row["target"],
                    session=row["session"] or None,
                    seed=int(row["seed"]),
                    run_directory=(
                        relative if relative.is_absolute() else REPOSITORY_ROOT / relative
                    ),
                    study=row.get("study") or "canonical",
                    sequence_length=int(row.get("sequence_length") or 24),
                    evaluation_scheme=row.get("evaluation_scheme") or "within-session",
                    held_out_session=row.get("held_out_session") or None,
                    reuse_canonical=(row.get("reuse_canonical", "").lower() == "true"),
                )
            )
    return runs


def required_diagnostics(run: ManifestRun) -> list[Path]:
    if run.study != "canonical":
        return []
    if run.target in {"area", "distance"}:
        return [
            run.run_directory / f"diagnostics_{run.target}_{slug}.png"
            for slug in POSITION_SESSION_SLUGS
        ]
    return [run.run_directory / "diagnostics.png"]


def validate_run_manifest(
    path: Path = MANIFEST_PATH,
    require_outputs: bool = False,
) -> list[ManifestRun]:
    runs = load_run_manifest(path)
    studies = {run.study for run in runs}
    if len(studies) != 1:
        raise ValueError(f"Run manifest must contain one study, found {sorted(studies)}")
    study = next(iter(studies))
    expected_counts = {"canonical": 180, "sequence-length": 40, "leave-one-session-out": 50}
    if study not in expected_counts:
        raise ValueError(f"Unknown manifest study: {study}")
    if len(runs) != expected_counts[study]:
        raise ValueError(
            f"{study} manifest must have exactly {expected_counts[study]} rows, "
            f"found {len(runs)}"
        )
    ids = [run.run_id for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Run manifest contains duplicate run IDs")
    if study == "canonical":
        configurations = [
            (run.architecture, run.temporal_mode, run.target, run.session, run.seed)
            for run in runs
        ]
        expected = expected_configurations()
    elif study == "sequence-length":
        configurations = [
            (
                run.architecture,
                run.temporal_mode,
                run.target,
                run.sequence_length,
                run.seed,
            )
            for run in runs
        ]
        expected = expected_sequence_length_configurations()
    else:
        configurations = [
            (
                run.architecture,
                run.temporal_mode,
                run.target,
                str(run.held_out_session),
                run.seed,
            )
            for run in runs
        ]
        expected = expected_loso_configurations()
    if len(configurations) != len(set(configurations)):
        raise ValueError("Run manifest contains duplicate configurations")
    observed = set(configurations)
    if observed != expected:
        missing = sorted(expected - observed, key=str)
        extra = sorted(observed - expected, key=str)
        raise ValueError(f"Run manifest mismatch; missing={missing}, extra={extra}")
    if study == "canonical":
        invalid = [
            run.run_id
            for run in runs
            if run.sequence_length != 24
            or run.evaluation_scheme != "within-session"
            or run.held_out_session is not None
            or run.reuse_canonical
        ]
    elif study == "sequence-length":
        invalid = [
            run.run_id
            for run in runs
            if run.session is not None
            or run.evaluation_scheme != "within-session"
            or run.held_out_session is not None
            or run.reuse_canonical != (run.sequence_length == 24)
        ]
    else:
        invalid = [
            run.run_id
            for run in runs
            if run.session is not None
            or run.evaluation_scheme != "leave-one-session-out"
            or run.held_out_session is None
            or run.sequence_length != 24
            or run.reuse_canonical
        ]
    if invalid:
        raise ValueError(f"Manifest rows violate the {study} protocol: {invalid}")
    directories = [run.run_directory.resolve() for run in runs]
    if len(directories) != len(set(directories)):
        raise ValueError("Run manifest maps multiple configurations to one directory")
    if require_outputs:
        from .training import fixed_topology

        for run in runs:
            required = [
                run.checkpoint_path,
                run.run_directory / "metrics.json",
                run.run_directory / "history.csv",
                run.run_directory / "aggregated_predictions.npz",
                run.run_directory / "resolved_config.json",
                run.run_directory / "data_signature.json",
                run.run_directory / "split_summary.json",
                *required_diagnostics(run),
            ]
            absent = [str(item) for item in required if not item.is_file()]
            if absent:
                raise FileNotFoundError(
                    f"Run {run.run_id} is incomplete; missing: {', '.join(absent)}"
                )
            checkpoint = torch.load(
                run.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            expected_config = {
                "architecture": run.architecture,
                "temporal_mode": run.temporal_mode,
                "target": run.target,
                "session": run.session,
                "seed": run.seed,
                "sequence_length": run.sequence_length,
                "evaluation_scheme": run.evaluation_scheme,
                "held_out_session": run.held_out_session,
            }
            actual = {
                key: checkpoint["config"].get(key) for key in expected_config
            }
            if actual != expected_config:
                raise ValueError(
                    f"Run {run.run_id} checkpoint configuration is incompatible: {actual}"
                )
            if checkpoint.get("signature", {}).get("fixed_topology") != fixed_topology(
                run.sequence_length
            ):
                raise ValueError(f"Run {run.run_id} checkpoint topology is incompatible")
            if (
                checkpoint.get("signature", {}).get("normalization_method")
                != "unique-training-physical-rows"
            ):
                raise ValueError(f"Run {run.run_id} uses incompatible normalization")
            with (run.run_directory / "metrics.json").open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            if not metrics.get("exploratory_validation_only"):
                raise ValueError(f"Run {run.run_id} is missing the validation-only marker")
    return runs


def representative_position_runs(runs: list[ManifestRun]) -> list[ManifestRun]:
    """Choose the seed nearest each five-seed median position Top-8."""
    selected: list[ManifestRun] = []
    grouped: dict[tuple[str, str, str], list[tuple[ManifestRun, float]]] = {}
    for run in runs:
        if run.target != "position":
            continue
        with (run.run_directory / "metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        value = float(metrics["equal_session_macro"]["top_8"])
        grouped.setdefault(
            (run.architecture, run.temporal_mode, str(run.session)), []
        ).append((run, value))
    for key, candidates in grouped.items():
        if {run.seed for run, _ in candidates} != set(SEEDS):
            raise ValueError(f"Incomplete seed group for representative selection: {key}")
        median = sorted(value for _, value in candidates)[len(SEEDS) // 2]
        selected.append(
            min(candidates, key=lambda item: (abs(item[1] - median), item[0].seed))[0]
        )
    if len(selected) != 20:
        raise ValueError(f"Expected 20 representative position runs, found {len(selected)}")
    return sorted(selected, key=lambda run: run.run_id)

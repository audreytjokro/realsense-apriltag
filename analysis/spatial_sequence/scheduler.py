from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from .manifest import (
    MANIFEST_PATH,
    OUTPUT_ROOT,
    ManifestRun,
    required_diagnostics,
    validate_run_manifest,
)


def _latest_checkpoint(run: ManifestRun) -> Path | None:
    candidates = sorted(run.run_directory.glob("epoch_*.pt"))
    if (run.run_directory / "final.pt").is_file():
        return run.run_directory / "final.pt"
    return candidates[-1] if candidates else None


def _run_complete(run: ManifestRun) -> bool:
    required = [
        run.run_directory / "best.pt",
        run.run_directory / "final.pt",
        run.run_directory / "history.csv",
        run.run_directory / "metrics.json",
        run.run_directory / "aggregated_predictions.npz",
        run.run_directory / "resolved_config.json",
        run.run_directory / "data_signature.json",
        *required_diagnostics(run),
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        configuration = json.loads(
            (run.run_directory / "resolved_config.json").read_text(encoding="utf-8")
        )
        signature = json.loads(
            (run.run_directory / "data_signature.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return False
    expected = {
        "architecture": run.architecture,
        "temporal_mode": run.temporal_mode,
        "target": run.target,
        "session": run.session,
        "seed": run.seed,
        "sequence_length": run.sequence_length,
        "evaluation_scheme": run.evaluation_scheme,
        "held_out_session": run.held_out_session,
    }
    return (
        all(configuration.get(key) == value for key, value in expected.items())
        and signature.get("normalization_method") == "unique-training-physical-rows"
    )


def _command_for_run(run: ManifestRun) -> list[str]:
    checkpoint = _latest_checkpoint(run)
    if checkpoint is not None and (run.run_directory / "best.pt").is_file():
        if checkpoint.name == "final.pt":
            return [
                sys.executable,
                "-m",
                "analysis.spatial_sequence",
                "--eval-only",
                "--checkpoint",
                str(run.run_directory / "best.pt"),
                "--output-directory",
                str(run.run_directory),
            ]
    command = [
        sys.executable,
        "-m",
        "analysis.spatial_sequence",
        "--architecture",
        run.architecture,
        "--temporal-mode",
        run.temporal_mode,
        "--target",
        run.target,
        "--seed",
        str(run.seed),
        "--epochs",
        "100",
        "--device",
        "cuda:0",
        "--output-directory",
        str(run.run_directory),
    ]
    if run.session is not None:
        command.extend(("--session", run.session))
    if run.sequence_length != 24:
        command.extend(("--sequence-length", str(run.sequence_length)))
    if run.evaluation_scheme != "within-session":
        command.extend(("--evaluation-scheme", run.evaluation_scheme))
    if run.held_out_session is not None:
        command.extend(("--held-out-session", run.held_out_session))
    if checkpoint is not None:
        command.extend(("--resume", str(checkpoint)))
    return command


def run_dynamic_gpu_queue(
    gpu_ids: list[int],
    manifest_path: Path = MANIFEST_PATH,
    excluded_run_ids: set[str] | None = None,
    log_directory: Path | None = None,
) -> list[tuple[str, int]]:
    """Run one process per GPU, dynamically claiming unfinished manifest rows."""
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("At least one unique GPU ID is required")
    runs = validate_run_manifest(manifest_path)
    excluded = excluded_run_ids or set()
    pending: queue.Queue[ManifestRun] = queue.Queue()
    for run in runs:
        if run.run_id in excluded:
            continue
        if run.reuse_canonical:
            if not _run_complete(run):
                raise FileNotFoundError(
                    f"Canonical dependency is incomplete or incompatible: {run.run_id}"
                )
            continue
        if not _run_complete(run):
            pending.put(run)
    log_directory = log_directory or (OUTPUT_ROOT / "logs")
    log_directory.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, int]] = []
    lock = threading.Lock()

    def worker(gpu_id: int) -> None:
        while True:
            try:
                run = pending.get_nowait()
            except queue.Empty:
                return
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            environment.setdefault("TMPDIR", "/dev/shm")
            log_path = log_directory / f"{run.run_id}.log"
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\nGPU {gpu_id}: {' '.join(_command_for_run(run))}\n")
                log.flush()
                completed = subprocess.run(
                    _command_for_run(run),
                    cwd=Path(__file__).resolve().parents[2],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode:
                with lock:
                    failures.append((run.run_id, completed.returncode))
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu_id,)) for gpu_id in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(failures)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the manifest on a dynamic GPU queue")
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU IDs")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--log-directory", type=Path)
    parser.add_argument("--exclude-run-id", action="append", default=[])
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    gpu_ids = [int(value) for value in arguments.gpus.split(",")]
    failures = run_dynamic_gpu_queue(
        gpu_ids,
        arguments.manifest,
        set(arguments.exclude_run_id),
        arguments.log_directory,
    )
    if failures:
        for run_id, return_code in failures:
            print(f"FAILED {run_id}: exit {return_code}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

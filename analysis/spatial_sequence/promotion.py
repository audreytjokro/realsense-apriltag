from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .manifest import MANIFEST_PATH, load_run_manifest
from .training import rewrite_checkpoint_metadata_atomic


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".migrating")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        verified = json.loads(temporary.read_text(encoding="utf-8"))
        if verified.keys() != payload.keys():
            raise ValueError(f"JSON rewrite verification failed: {path}")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def rewrite_staged_output_paths(
    staging_manifest: Path,
    canonical_manifest: Path = MANIFEST_PATH,
) -> int:
    """Prepare staged canonical runs for an atomic directory promotion.

    Only path metadata changes. Every checkpoint rewrite uses the verified atomic
    helper, which confirms that all non-replaced training state is identical.
    """
    staged = {run.run_id: run for run in load_run_manifest(staging_manifest)}
    canonical = {run.run_id: run for run in load_run_manifest(canonical_manifest)}
    if staged.keys() != canonical.keys():
        raise ValueError("Staging and canonical manifests contain different run IDs")

    rewritten_checkpoints = 0
    for index, run_id in enumerate(sorted(staged), start=1):
        source = staged[run_id]
        destination = canonical[run_id].run_directory.resolve()
        destination_text = str(destination)

        checkpoint_paths = sorted(source.run_directory.glob("*.pt"))
        if len(checkpoint_paths) != 22:
            raise ValueError(
                f"Expected 22 checkpoints for {run_id}, found {len(checkpoint_paths)}"
            )
        for checkpoint_path in checkpoint_paths:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = dict(payload["config"])
            if config.get("output_directory") == destination_text:
                continue
            config["output_directory"] = destination_text
            rewrite_checkpoint_metadata_atomic(
                checkpoint_path,
                {"config": config},
            )
            rewritten_checkpoints += 1

        config_path = source.run_directory / "resolved_config.json"
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        config_payload["output_directory"] = destination_text
        _write_json_atomic(config_path, config_payload)

        metrics_path = source.run_directory / "metrics.json"
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_payload["checkpoint"] = str(destination / "best.pt")
        _write_json_atomic(metrics_path, metrics_payload)

        if index % 10 == 0 or index == len(staged):
            print(
                f"prepared_runs={index}/{len(staged)} "
                f"rewritten_checkpoints={rewritten_checkpoints}",
                flush=True,
            )
    return rewritten_checkpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite staged canonical path metadata before promotion."
    )
    parser.add_argument("--staging-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, default=MANIFEST_PATH)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    count = rewrite_staged_output_paths(
        arguments.staging_manifest,
        arguments.canonical_manifest,
    )
    print(f"checkpoint_rewrites_complete={count}")


if __name__ == "__main__":
    main()

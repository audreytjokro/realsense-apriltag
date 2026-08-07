"""Spatial ground-truth/prediction figures for the GRU LOSO area runs, in the identical
style/layout to the existing TCN figures (analysis/spatial_sequence/visualization.py's
render_cross_seed_diagnostics, reused unchanged -- not reimplemented).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.spatial_sequence.core import REPOSITORY_ROOT, discover_sessions, load_annotation
from analysis.spatial_sequence.visualization import render_cross_seed_diagnostics

from .report import SEEDS, SESSIONS, _run_directory


def _load_aggregate(held_out_session: str, seed: int, architecture_version: str) -> dict[str, np.ndarray]:
    directory = _run_directory(held_out_session, seed, architecture_version)
    with np.load(directory / "aggregated_predictions.npz") as record:
        return {name: record[name].copy() for name in record.files}


def generate_area_figures(architecture_version: str = "v2") -> list[Path]:
    session_layouts = {
        info.slug: load_annotation(info).source_polygons_paper_cm for info in discover_sessions()
    }
    architecture = "bigru" if architecture_version == "v1" else "bigru-v2"
    output_directory = REPOSITORY_ROOT / "analysis" / "reports" / "gru-sequence" / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for session in SESSIONS:
        # _session_slug is passed explicitly below, which skips the function's own
        # per-session auto-suffixing (that only triggers when _session_slug is None)
        # -- so the session name has to be in the path we hand it, per-call.
        output_path = output_directory / f"area_bidirectional_{architecture}_{session}.png"
        aggregates = [_load_aggregate(session, seed, architecture_version) for seed in SEEDS]
        produced = render_cross_seed_diagnostics(
            "area",
            aggregates,
            output_path,
            architecture=architecture,
            temporal_mode="bidirectional",
            session_layouts=session_layouts,
            constrained_layout=False,
            _session_slug=session,
        )
        outputs.extend(produced)
    if len(outputs) != len(SESSIONS):
        raise AssertionError(f"Expected {len(SESSIONS)} figures, produced {len(outputs)}")
    return outputs


if __name__ == "__main__":
    for path in generate_area_figures():
        print(path)

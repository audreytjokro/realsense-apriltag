"""Reading a recorded session's synchronized CSV and sidecar JSON files.

The CSV column names and JSON filenames here are fixed by
record_cyranose_reading_pose.py's output format, not by TrialConfig: any
session recorded with that recorder, on any desk layout or odor, has this
same file/column shape. Only the physical parameters (lag, height band, desk
bounds) vary per trial, which is what TrialConfig is for.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionData:
    session_directory: Path
    raw_rows: list[dict[str, str]]
    metadata: dict[str, Any]
    alignment: dict[str, Any]


def find_session_by_trial_id(root: Path, trial_id: str) -> Path:
    """Find the single session directory under `root` whose recorded trial_id matches.

    Raises if zero or more than one session matches, since analysis code should
    never silently guess which recording it means.
    """
    matches = []
    for metadata_path in root.rglob("session_metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("trial_id") == trial_id:
            matches.append(metadata_path.parent)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one session for trial_id {trial_id!r}; found "
            f"{len(matches)}: {matches}"
        )
    return matches[0]


def load_session(session_directory: Path) -> SessionData:
    """Read a session's synchronized CSV, metadata, and alignment summary."""
    csv_path = session_directory / "cyranose_reading_pose.csv"
    with csv_path.open(newline="", encoding="utf-8") as source:
        raw_rows = list(csv.DictReader(source))
    metadata = json.loads(
        (session_directory / "session_metadata.json").read_text(encoding="utf-8")
    )
    alignment = json.loads(
        (session_directory / "alignment_summary.json").read_text(encoding="utf-8")
    )
    return SessionData(
        session_directory=session_directory,
        raw_rows=raw_rows,
        metadata=metadata,
        alignment=alignment,
    )

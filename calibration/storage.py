from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

RECORDING_PATH = DATA_DIR / "recording.mp4"
ADDITIONAL_RECORDINGS_DIR = DATA_DIR / "recordings"
CAMERA_INTRINSICS_PATH = DATA_DIR / "camera_intrinsics.json"
COVERAGE_PATH = DATA_DIR / "coverage.json"
CUBE_CALIBRATION_PATH = DATA_DIR / "cube_calibration.json"
SNOUT_CALIBRATION_PATH = DATA_DIR / "snout_calibration.json"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def recording_paths() -> list[Path]:
    """Return the legacy primary recording followed by all added clips."""
    paths: list[Path] = []
    if RECORDING_PATH.exists():
        paths.append(RECORDING_PATH)
    if ADDITIONAL_RECORDINGS_DIR.exists():
        paths.extend(sorted(ADDITIONAL_RECORDINGS_DIR.glob("*.mp4")))
    return paths


def new_additional_recording_path(now: datetime | None = None) -> Path:
    """Create a collision-resistant path for a non-destructive extra clip."""
    ensure_data_dir()
    ADDITIONAL_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    return ADDITIONAL_RECORDINGS_DIR / f"clip_{timestamp}.mp4"


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def save_json(path: Path, data: dict[str, Any]) -> None:
    ensure_data_dir()
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_value(data), file, indent=2)
        file.write("\n")


def load_camera_intrinsics() -> dict[str, Any] | None:
    return load_json_if_exists(CAMERA_INTRINSICS_PATH)


def save_camera_intrinsics(data: dict[str, Any]) -> None:
    save_json(CAMERA_INTRINSICS_PATH, data)


def load_cube_calibration() -> dict[str, Any] | None:
    return load_json_if_exists(CUBE_CALIBRATION_PATH)


def save_cube_calibration(data: dict[str, Any]) -> None:
    save_json(CUBE_CALIBRATION_PATH, data)


def load_snout_calibration() -> dict[str, Any] | None:
    return load_json_if_exists(SNOUT_CALIBRATION_PATH)


def save_snout_calibration(data: dict[str, Any]) -> None:
    save_json(SNOUT_CALIBRATION_PATH, data)


def save_coverage(data: dict[str, Any]) -> None:
    save_json(COVERAGE_PATH, data)

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
SESSION_PREFIX = "cyranose_reading_pose_session"
SESSION_METADATA_NAME = "session_metadata.json"
RECORDER_CSV_NAME = "cyranose_reading_pose.csv"
WAYPOINT_CSV_NAME = "waypoint_sequence.csv"
WAYPOINT_METADATA_NAME = "waypoint_metadata.json"
WINDOW_NAME = "Optional Random Waypoint Guide"

DEFAULT_WIDTH_CM = 26.5
DEFAULT_HEIGHT_CM = 26.5
DEFAULT_MARGIN_CM = 1.5
DEFAULT_DURATION_S = 600.0
DEFAULT_SESSION_WAIT_S = 30.0
DEFAULT_MAX_SESSION_AGE_S = 3600.0

EVENT_FIELDS = [
    "event_index",
    "event",
    "waypoint_index",
    "x_cm",
    "y_cm",
    "next_x_cm",
    "next_y_cm",
    "host_timestamp_utc",
    "recorder_elapsed_s",
]


@dataclass(frozen=True)
class Waypoint:
    index: int
    x_cm: float
    y_cm: float


class WaypointGenerator:
    def __init__(
        self,
        seed: int,
        width_cm: float,
        height_cm: float,
        margin_cm: float,
    ) -> None:
        if width_cm <= 0 or height_cm <= 0:
            raise ValueError("Paper width and height must be positive")
        if margin_cm < 0:
            raise ValueError("--margin-cm must be non-negative")
        if 2 * margin_cm >= min(width_cm, height_cm):
            raise ValueError("--margin-cm leaves no usable waypoint area")

        self.random = random.Random(seed)
        self.width_cm = width_cm
        self.height_cm = height_cm
        self.margin_cm = margin_cm
        self.next_index = 1

    def next(self) -> Waypoint:
        point = Waypoint(
            index=self.next_index,
            x_cm=round(
                self.random.uniform(
                    self.margin_cm,
                    self.width_cm - self.margin_cm,
                ),
                2,
            ),
            y_cm=round(
                self.random.uniform(
                    self.margin_cm,
                    self.height_cm - self.margin_cm,
                ),
                2,
            ),
        )
        self.next_index += 1
        return point


class EventLogger:
    def __init__(
        self,
        file: TextIO,
        recorder_elapsed_reader,
    ) -> None:
        self.file = file
        self.writer = csv.DictWriter(file, fieldnames=EVENT_FIELDS)
        self.writer.writeheader()
        self.file.flush()
        self.recorder_elapsed_reader = recorder_elapsed_reader
        self.event_index = 1

    def log(
        self,
        event: str,
        current: Waypoint | None,
        next_waypoint: Waypoint | None,
    ) -> None:
        elapsed_s = self.recorder_elapsed_reader()
        self.writer.writerow(
            {
                "event_index": self.event_index,
                "event": event,
                "waypoint_index": "" if current is None else current.index,
                "x_cm": "" if current is None else current.x_cm,
                "y_cm": "" if current is None else current.y_cm,
                "next_x_cm": ""
                if next_waypoint is None
                else next_waypoint.x_cm,
                "next_y_cm": ""
                if next_waypoint is None
                else next_waypoint.y_cm,
                "host_timestamp_utc": utc_now(),
                "recorder_elapsed_s": ""
                if elapsed_s is None
                else round(elapsed_s, 3),
            }
        )
        self.file.flush()
        self.event_index += 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def load_trial_id(session_dir: Path) -> str | None:
    metadata_path = session_dir / SESSION_METADATA_NAME
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    trial_id = metadata.get("trial_id")
    return str(trial_id) if trial_id is not None else None


def find_recent_session(
    trial_id: str,
    max_age_s: float,
    root_dir: Path = ROOT_DIR,
) -> Path | None:
    now = time.time()
    matches: list[Path] = []
    for candidate in root_dir.glob(f"{SESSION_PREFIX}_*"):
        if not candidate.is_dir():
            continue
        try:
            age_s = now - candidate.stat().st_mtime
        except OSError:
            continue
        if age_s > max_age_s:
            continue
        if load_trial_id(candidate) == trial_id:
            matches.append(candidate)
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def wait_for_session(
    trial_id: str,
    wait_s: float,
    max_age_s: float,
) -> Path:
    deadline = time.monotonic() + wait_s
    while True:
        session_dir = find_recent_session(trial_id, max_age_s)
        if session_dir is not None:
            return session_dir
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "No recent recorder session found with trial ID "
                f"{trial_id!r}. Start record_cyranose_reading_pose.py first "
                "using the same --trial-id."
            )
        time.sleep(0.25)


def resolve_session_dir(args: argparse.Namespace) -> Path:
    if args.session_dir is None:
        return wait_for_session(
            args.trial_id,
            args.wait_for_session_s,
            args.max_session_age_s,
        )

    session_dir = args.session_dir.resolve()
    if not session_dir.is_dir():
        raise FileNotFoundError(session_dir)
    recorded_trial_id = load_trial_id(session_dir)
    if recorded_trial_id != args.trial_id:
        raise ValueError(
            f"{session_dir} records trial ID {recorded_trial_id!r}, "
            f"not {args.trial_id!r}"
        )
    return session_dir


def read_latest_recorder_elapsed_s(csv_path: Path) -> float | None:
    try:
        with csv_path.open("rb") as file:
            header_bytes = file.readline()
            header_end = file.tell()
            if not header_bytes:
                return None
            header = next(
                csv.reader([header_bytes.decode("utf-8").rstrip("\r\n")])
            )
            timestamp_index = header.index("pcnose_sample_time_estimate_utc")

            first_data_line = file.readline()
            if not first_data_line:
                return None
            first_values = next(
                csv.reader([first_data_line.decode("utf-8").rstrip("\r\n")])
            )
            if timestamp_index >= len(first_values):
                return None
            first_timestamp = datetime.fromisoformat(
                first_values[timestamp_index].replace("Z", "+00:00")
            )

            file.seek(0, 2)
            file_size = file.tell()
            if file_size <= header_end:
                return None
            tail_start = max(header_end, file_size - 65_536)
            file.seek(tail_start)
            tail = file.read()
            lines = tail.splitlines()
            if tail and not tail.endswith((b"\n", b"\r")):
                lines = lines[:-1]
            if tail_start > header_end and lines:
                lines = lines[1:]

            for raw_line in reversed(lines):
                try:
                    values = next(csv.reader([raw_line.decode("utf-8")]))
                except (UnicodeDecodeError, csv.Error):
                    continue
                if timestamp_index >= len(values):
                    continue
                try:
                    latest_timestamp = datetime.fromisoformat(
                        values[timestamp_index].replace("Z", "+00:00")
                    )
                    return max(
                        0.0,
                        (latest_timestamp - first_timestamp).total_seconds(),
                    )
                except ValueError:
                    continue
            return None
    except (FileNotFoundError, OSError, ValueError, csv.Error):
        return None


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    total_seconds = max(0, int(seconds))
    minutes, remainder = divmod(total_seconds, 60)
    return f"{minutes:02d}:{remainder:02d}"


def paper_to_pixel(
    waypoint: Waypoint,
    width_cm: float,
    height_cm: float,
    left: int,
    top: int,
    size: int,
) -> tuple[int, int]:
    x = left + int(round(size * waypoint.x_cm / width_cm))
    y = top + int(round(size * waypoint.y_cm / height_cm))
    return x, y


def draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_guide(
    width_cm: float,
    height_cm: float,
    margin_cm: float,
    current: Waypoint,
    next_waypoint: Waypoint,
    completed: list[Waypoint],
    elapsed_s: float | None,
    duration_s: float,
    stream_is_fresh: bool,
) -> np.ndarray:
    image = np.full((720, 1080, 3), (30, 33, 38), dtype=np.uint8)
    paper_left = 55
    paper_top = 105
    paper_size = 550
    panel_left = 675

    draw_text(
        image,
        "OPTIONAL RANDOM WAYPOINT GUIDE",
        (55, 45),
        scale=0.9,
        thickness=2,
    )
    draw_text(
        image,
        "Paper origin is top-left; suggestions never control the recorder.",
        (55, 72),
        scale=0.55,
        color=(185, 190, 198),
    )

    cv2.rectangle(
        image,
        (paper_left, paper_top),
        (paper_left + paper_size, paper_top + paper_size),
        (215, 215, 215),
        thickness=-1,
    )
    margin_x = int(round(paper_size * margin_cm / width_cm))
    margin_y = int(round(paper_size * margin_cm / height_cm))
    cv2.rectangle(
        image,
        (paper_left + margin_x, paper_top + margin_y),
        (
            paper_left + paper_size - margin_x,
            paper_top + paper_size - margin_y,
        ),
        (120, 125, 132),
        thickness=2,
    )

    for point in completed[-20:]:
        x, y = paper_to_pixel(
            point,
            width_cm,
            height_cm,
            paper_left,
            paper_top,
            paper_size,
        )
        cv2.circle(image, (x, y), 4, (145, 145, 145), thickness=-1)

    next_x, next_y = paper_to_pixel(
        next_waypoint,
        width_cm,
        height_cm,
        paper_left,
        paper_top,
        paper_size,
    )
    cv2.circle(image, (next_x, next_y), 11, (225, 155, 70), thickness=2)
    next_label_x = next_x + 14
    if next_x > paper_left + paper_size - 105:
        next_label_x = next_x - 78
    draw_text(
        image,
        f"next {next_waypoint.index}",
        (next_label_x, next_y - 10),
        scale=0.45,
        color=(80, 85, 90),
    )

    current_x, current_y = paper_to_pixel(
        current,
        width_cm,
        height_cm,
        paper_left,
        paper_top,
        paper_size,
    )
    cv2.circle(image, (current_x, current_y), 13, (80, 185, 95), thickness=-1)
    cv2.circle(image, (current_x, current_y), 17, (55, 120, 65), thickness=2)
    current_label_x = current_x + 18
    if current_x > paper_left + paper_size - 125:
        current_label_x = current_x - 105
    draw_text(
        image,
        f"current {current.index}",
        (current_label_x, current_y + 6),
        scale=0.5,
        color=(45, 50, 55),
        thickness=2,
    )

    draw_text(image, "0 cm", (paper_left - 2, paper_top - 10), scale=0.45)
    draw_text(
        image,
        f"{width_cm:g} cm",
        (paper_left + paper_size - 70, paper_top - 10),
        scale=0.45,
    )
    draw_text(
        image,
        f"{height_cm:g} cm",
        (paper_left - 3, paper_top + paper_size + 25),
        scale=0.45,
    )

    remaining_s = None if elapsed_s is None else max(0.0, duration_s - elapsed_s)
    draw_text(image, "RECORDER ELAPSED", (panel_left, 125), scale=0.55)
    draw_text(
        image,
        format_duration(elapsed_s),
        (panel_left, 190),
        scale=1.8,
        thickness=3,
    )
    draw_text(image, "REMAINING", (panel_left, 235), scale=0.55)
    draw_text(
        image,
        format_duration(remaining_s),
        (panel_left, 290),
        scale=1.35,
        thickness=2,
    )

    status_text = "Recorder stream: updating" if stream_is_fresh else "Recorder stream: waiting/stale"
    status_color = (90, 200, 105) if stream_is_fresh else (90, 175, 235)
    draw_text(
        image,
        status_text,
        (panel_left, 335),
        scale=0.55,
        color=status_color,
        thickness=2,
    )

    draw_text(image, "CURRENT SUGGESTION", (panel_left, 390), scale=0.55)
    draw_text(
        image,
        f"X {current.x_cm:5.2f} cm",
        (panel_left, 430),
        scale=0.75,
        thickness=2,
    )
    draw_text(
        image,
        f"Y {current.y_cm:5.2f} cm",
        (panel_left, 465),
        scale=0.75,
        thickness=2,
    )
    draw_text(
        image,
        f"Next: ({next_waypoint.x_cm:.2f}, {next_waypoint.y_cm:.2f}) cm",
        (panel_left, 515),
        scale=0.55,
        color=(185, 190, 198),
    )

    draw_text(
        image,
        "SPACE or N: next suggestion",
        (panel_left, 580),
        scale=0.55,
    )
    draw_text(
        image,
        "Q or Esc: close guide only",
        (panel_left, 615),
        scale=0.55,
    )
    draw_text(
        image,
        "You may ignore every suggestion.",
        (panel_left, 650),
        scale=0.55,
        color=(185, 190, 198),
    )

    if elapsed_s is not None and elapsed_s >= duration_s:
        cv2.rectangle(image, (655, 20), (1060, 78), (40, 40, 205), thickness=-1)
        draw_text(
            image,
            "10 MINUTES REACHED - STOP RECORDER",
            (675, 58),
            scale=0.65,
            thickness=2,
        )

    return image


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    session_dir: Path,
) -> None:
    metadata = {
        "created_at_utc": utc_now(),
        "trial_id": args.trial_id,
        "recorder_session_dir": str(session_dir),
        "paper_width_cm": args.width_cm,
        "paper_height_cm": args.height_cm,
        "margin_cm": args.margin_cm,
        "random_seed": args.seed,
        "guide_duration_s": args.duration_s,
        "elapsed_time_basis": (
            "host serial-request midpoint UTC delta "
            "(pcnose_sample_time_estimate_utc)"
        ),
        "advance_keys": ["space", "n"],
        "guide_controls_recorder": False,
        "coordinate_origin": "paper_top_left",
    }
    with path.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")


def run_guide(args: argparse.Namespace) -> None:
    session_dir = resolve_session_dir(args)
    recorder_csv = session_dir / RECORDER_CSV_NAME
    waypoint_csv = session_dir / WAYPOINT_CSV_NAME
    waypoint_metadata = session_dir / WAYPOINT_METADATA_NAME
    if waypoint_csv.exists() or waypoint_metadata.exists():
        raise FileExistsError(
            "Waypoint output already exists in this session; refusing to overwrite it"
        )

    generator = WaypointGenerator(
        args.seed,
        args.width_cm,
        args.height_cm,
        args.margin_cm,
    )
    current = generator.next()
    next_waypoint = generator.next()
    completed: list[Waypoint] = []

    write_metadata(waypoint_metadata, args, session_dir)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1080, 720)

    print(f"Attached to recorder session: {session_dir}")
    print(f"Writing waypoint events to: {waypoint_csv}")
    print("SPACE or N advances the optional suggestion.")
    print("Q, Esc, or closing this window stops only the guide.")
    print("The Cyranose + RealSense recorder must still be stopped manually.")

    duration_event_logged = False
    last_elapsed_check = 0.0
    elapsed_s: float | None = None
    stream_is_fresh = False

    with waypoint_csv.open("x", newline="", encoding="utf-8") as event_file:
        logger = EventLogger(
            event_file,
            lambda: read_latest_recorder_elapsed_s(recorder_csv),
        )
        logger.log("guide_attached", current, next_waypoint)
        logger.log("waypoint_shown", current, next_waypoint)

        try:
            while True:
                now = time.monotonic()
                if now - last_elapsed_check >= 0.2:
                    elapsed_s = read_latest_recorder_elapsed_s(recorder_csv)
                    try:
                        stream_is_fresh = (
                            time.time() - recorder_csv.stat().st_mtime
                        ) <= 3.0
                    except OSError:
                        stream_is_fresh = False
                    last_elapsed_check = now

                if (
                    elapsed_s is not None
                    and elapsed_s >= args.duration_s
                    and not duration_event_logged
                ):
                    logger.log("duration_reached", current, next_waypoint)
                    duration_event_logged = True
                    print(
                        f"{args.duration_s:g} seconds reached. "
                        "Stop the recorder manually."
                    )

                image = draw_guide(
                    args.width_cm,
                    args.height_cm,
                    args.margin_cm,
                    current,
                    next_waypoint,
                    completed,
                    elapsed_s,
                    args.duration_s,
                    stream_is_fresh,
                )
                cv2.imshow(WINDOW_NAME, image)
                key = cv2.waitKey(50) & 0xFF
                if key in (ord(" "), ord("n"), ord("N")):
                    logger.log("waypoint_advanced", current, next_waypoint)
                    completed.append(current)
                    current = next_waypoint
                    next_waypoint = generator.next()
                    logger.log("waypoint_shown", current, next_waypoint)
                elif key in (ord("q"), ord("Q"), 27):
                    break

                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        finally:
            logger.log("guide_closed", current, next_waypoint)
            cv2.destroyWindow(WINDOW_NAME)

    print(f"Saved waypoint sequence and timestamps to: {waypoint_csv}")
    print(f"Saved waypoint settings to: {waypoint_metadata}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show optional reproducible random paper waypoints and log their "
            "timestamps beside an active Cyranose + RealSense recording."
        )
    )
    parser.add_argument(
        "--trial-id",
        required=True,
        help="Exact trial ID passed to record_cyranose_reading_pose.py.",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--width-cm", type=float, default=DEFAULT_WIDTH_CM)
    parser.add_argument("--height-cm", type=float, default=DEFAULT_HEIGHT_CM)
    parser.add_argument("--margin-cm", type=float, default=DEFAULT_MARGIN_CM)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=DEFAULT_DURATION_S,
        help=(
            "Guide timer duration. Reaching it displays an alert but does not "
            "stop the recorder."
        ),
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        help=(
            "Optional explicit recorder session folder. By default the guide "
            "finds the newest recent root session with the same trial ID."
        ),
    )
    parser.add_argument(
        "--wait-for-session-s",
        type=float,
        default=DEFAULT_SESSION_WAIT_S,
    )
    parser.add_argument(
        "--max-session-age-s",
        type=float,
        default=DEFAULT_MAX_SESSION_AGE_S,
    )
    args = parser.parse_args(argv)
    if args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    if args.wait_for_session_s < 0:
        parser.error("--wait-for-session-s must be non-negative")
    if args.max_session_age_s <= 0:
        parser.error("--max-session-age-s must be positive")
    return args


def main() -> None:
    try:
        run_guide(parse_args())
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()

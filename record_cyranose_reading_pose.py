from __future__ import annotations

import argparse
import csv
import json
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import pcnose_serial
import track_calibrated_cyranose_pose as pose_tracking


ROOT_DIR = Path(__file__).resolve().parent
WINDOW_NAME = "Cyranose Reading + Pose"
SESSION_PREFIX = "cyranose_reading_pose_session"
CSV_NAME = "cyranose_reading_pose.csv"
VIDEO_NAME = pose_tracking.VIDEO_NAME
VIDEO_CODEC = pose_tracking.VIDEO_CODEC
DEFAULT_PORT = "COM4"
DEFAULT_BAUD = 57600
DEFAULT_INTERVAL_S = 0.2
DEFAULT_MAX_SYNC_MS = 250.0
RESPONSE_TIMEOUT_S = 2.0
RESPONSE_ATTEMPTS = 2
ALIGNMENT_SUMMARY_NAME = "alignment_summary.json"
SESSION_METADATA_NAME = "session_metadata.json"

PCNOSE_SOURCE_FIELDS = pcnose_serial.decoded_field_names(include_raw_frame=True)
PCNOSE_FIELDS = [f"pcnose_{field}" for field in PCNOSE_SOURCE_FIELDS]
POSE_FIELDS = [
    "pose_host_timestamp_utc",
    "pose_elapsed_s",
    *pose_tracking.CSV_FIELDS[2:],
]
ALIGNMENT_FIELDS = [
    "pcnose_sample_time_estimate_utc",
    "pcnose_timestamp_basis",
    "pcnose_serial_roundtrip_ms",
    "pose_alignment_valid",
    "pose_alignment_status",
    "pose_minus_pcnose_ms",
    "abs_pose_minus_pcnose_ms",
]
CSV_FIELDS = [*PCNOSE_FIELDS, *ALIGNMENT_FIELDS, *POSE_FIELDS]


@dataclass(frozen=True)
class CyranoseReading:
    monotonic_ns: int
    sample_time_estimate_utc: str
    serial_roundtrip_ms: float
    decoded: dict[str, object]


@dataclass(frozen=True)
class PoseSample:
    monotonic_ns: int
    row: dict[str, Any]
    camera_frame: pose_tracking.RectifiedColorFrame
    observations: list[pose_tracking.geometry.TagObservation]
    T_camera_cube: np.ndarray | None
    cube_rms_px: float
    used_tag_ids: tuple[int, ...]


@dataclass
class AlignmentStats:
    max_sync_ms: float
    signed_deltas_ms: list[float]
    serial_roundtrips_ms: list[float]
    accepted: int = 0
    rejected: int = 0

    @classmethod
    def create(cls, max_sync_ms: float) -> "AlignmentStats":
        return cls(max_sync_ms, [], [])

    def add(self, signed_delta_ms: float, serial_roundtrip_ms: float) -> bool:
        valid = abs(signed_delta_ms) <= self.max_sync_ms
        self.signed_deltas_ms.append(signed_delta_ms)
        self.serial_roundtrips_ms.append(serial_roundtrip_ms)
        if valid:
            self.accepted += 1
        else:
            self.rejected += 1
        return valid

    @property
    def total(self) -> int:
        return self.accepted + self.rejected

    def as_dict(self) -> dict[str, Any]:
        absolute_deltas = [abs(value) for value in self.signed_deltas_ms]
        return {
            "timestamp_basis": "serial_request_midpoint",
            "max_sync_ms": self.max_sync_ms,
            "total_readings": self.total,
            "matched_readings": self.accepted,
            "rejected_readings": self.rejected,
            "match_rate_percent": round(
                100.0 * self.accepted / self.total,
                3,
            )
            if self.total
            else None,
            "signed_pose_minus_pcnose_ms": summarize_values(
                self.signed_deltas_ms,
            ),
            "absolute_pose_minus_pcnose_ms": summarize_values(
                absolute_deltas,
            ),
            "serial_roundtrip_ms": summarize_values(
                self.serial_roundtrips_ms,
            ),
        }


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "median": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": round(float(np.median(array)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "p99": round(float(np.percentile(array, 99)), 6),
        "maximum": round(float(np.max(array)), 6),
    }


class CyranoseReaderThread(threading.Thread):
    def __init__(self, port: str, baud: int, interval_s: float) -> None:
        super().__init__(name="cyranose-reader", daemon=True)
        self.port = port
        self.baud = baud
        self.interval_s = interval_s
        self.readings: queue.Queue[CyranoseReading] = queue.Queue()
        self.ready = threading.Event()
        self.start_recording = threading.Event()
        self.stop_requested = threading.Event()
        self.probe_response: str | None = None
        self.error: Exception | None = None

    def run(self) -> None:
        first_tick: int | None = None
        try:
            with pcnose_serial.WinSerial(self.port, self.baud) as serial:
                self.probe_response, _ = pcnose_serial.request_frame(
                    serial,
                    "RVN",
                    ["VN"],
                    timeout_s=RESPONSE_TIMEOUT_S,
                    attempts=1,
                )
                self.ready.set()

                while not self.start_recording.is_set():
                    if self.stop_requested.wait(0.05):
                        return

                while not self.stop_requested.is_set():
                    request_started_ns = time.perf_counter_ns()
                    request_started_utc = datetime.now(timezone.utc)
                    line, _ = pcnose_serial.request_frame(
                        serial,
                        "RSD",
                        ["SD"],
                        timeout_s=RESPONSE_TIMEOUT_S,
                        attempts=RESPONSE_ATTEMPTS,
                    )
                    received_ns = time.perf_counter_ns()
                    received_utc = datetime.now(timezone.utc)
                    sample_estimate_ns = (request_started_ns + received_ns) // 2
                    sample_estimate_utc = request_started_utc + (
                        received_utc - request_started_utc
                    ) / 2
                    serial_roundtrip_ms = (
                        received_ns - request_started_ns
                    ) / 1_000_000.0
                    frame = pcnose_serial.parse_sd_frame(line)
                    tick = frame.device_tick_deciseconds
                    if first_tick is None and tick is not None:
                        first_tick = tick
                    decoded = frame.as_decoded_row(
                        host_time_utc=received_utc.isoformat(timespec="milliseconds"),
                        first_tick_deciseconds=first_tick,
                        include_raw_frame=True,
                    )
                    self.readings.put(
                        CyranoseReading(
                            monotonic_ns=sample_estimate_ns,
                            sample_time_estimate_utc=sample_estimate_utc.isoformat(
                                timespec="milliseconds"
                            ),
                            serial_roundtrip_ms=serial_roundtrip_ms,
                            decoded=decoded,
                        )
                    )
                    if self.stop_requested.wait(self.interval_s):
                        return
        except Exception as exc:
            if not self.stop_requested.is_set():
                self.error = exc
        finally:
            self.ready.set()

    def request_stop(self) -> None:
        self.stop_requested.set()
        self.start_recording.set()


def session_paths(start_time: datetime) -> tuple[Path, Path]:
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT_DIR / f"{SESSION_PREFIX}_{timestamp}"
    return output_dir, output_dir / CSV_NAME


def rename_pose_metadata(row: dict[str, Any]) -> dict[str, Any]:
    renamed = dict(row)
    renamed["pose_host_timestamp_utc"] = renamed.pop("host_timestamp_utc")
    renamed["pose_elapsed_s"] = renamed.pop("elapsed_s")
    return renamed


def build_pose_sample(
    session_start_ns: int,
    camera_frame: pose_tracking.RectifiedColorFrame,
    pose_monotonic_ns: int,
    pose_timestamp_utc: str,
    camera_matrix: np.ndarray,
    T_cube_tag: dict[int, np.ndarray],
    p_snout_cube_mm: np.ndarray,
) -> PoseSample:
    observations = pose_tracking.geometry.detect_tags(
        camera_frame.image,
        camera_matrix,
    )
    pose_row, T_camera_cube, cube_rms, used_tag_ids = pose_tracking.build_csv_row(
        pose_timestamp_utc,
        (pose_monotonic_ns - session_start_ns) / 1_000_000_000.0,
        camera_frame,
        observations,
        camera_matrix,
        T_cube_tag,
        p_snout_cube_mm,
    )
    return PoseSample(
        monotonic_ns=pose_monotonic_ns,
        row=rename_pose_metadata(pose_row),
        camera_frame=camera_frame,
        observations=observations,
        T_camera_cube=T_camera_cube,
        cube_rms_px=cube_rms,
        used_tag_ids=used_tag_ids,
    )


def capture_pose_sample(
    camera: pose_tracking.RectifiedColorCamera,
    session_start_ns: int,
    camera_matrix: np.ndarray,
    T_cube_tag: dict[int, np.ndarray],
    p_snout_cube_mm: np.ndarray,
) -> PoseSample:
    camera_frame = camera.read_frame()
    pose_monotonic_ns = time.perf_counter_ns()
    pose_timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    return build_pose_sample(
        session_start_ns,
        camera_frame,
        pose_monotonic_ns,
        pose_timestamp_utc,
        camera_matrix,
        T_cube_tag,
        p_snout_cube_mm,
    )


def select_nearest_pose(
    reading: CyranoseReading,
    previous_pose: PoseSample | None,
    current_pose: PoseSample,
) -> PoseSample:
    if previous_pose is None:
        return current_pose
    previous_delta = abs(reading.monotonic_ns - previous_pose.monotonic_ns)
    current_delta = abs(current_pose.monotonic_ns - reading.monotonic_ns)
    return previous_pose if previous_delta <= current_delta else current_pose


def drain_readings(
    source: queue.Queue[CyranoseReading],
    pending: deque[CyranoseReading],
) -> None:
    while True:
        try:
            pending.append(source.get_nowait())
        except queue.Empty:
            return


def match_ready_readings(
    pending: deque[CyranoseReading],
    previous_pose: PoseSample | None,
    current_pose: PoseSample,
) -> list[tuple[CyranoseReading, PoseSample]]:
    matched: list[tuple[CyranoseReading, PoseSample]] = []
    while pending and pending[0].monotonic_ns <= current_pose.monotonic_ns:
        reading = pending.popleft()
        matched.append(
            (
                reading,
                select_nearest_pose(reading, previous_pose, current_pose),
            )
        )
    return matched


def build_combined_row(
    reading: CyranoseReading,
    pose: PoseSample,
    max_sync_ms: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        f"pcnose_{field}": reading.decoded[field]
        for field in PCNOSE_SOURCE_FIELDS
    }
    signed_delta_ms = (
        pose.monotonic_ns - reading.monotonic_ns
    ) / 1_000_000.0
    alignment_valid = abs(signed_delta_ms) <= max_sync_ms
    row.update(
        {
            "pcnose_sample_time_estimate_utc": reading.sample_time_estimate_utc,
            "pcnose_timestamp_basis": "serial_request_midpoint",
            "pcnose_serial_roundtrip_ms": round(
                reading.serial_roundtrip_ms,
                6,
            ),
            "pose_alignment_valid": alignment_valid,
            "pose_alignment_status": "matched"
            if alignment_valid
            else "over_threshold",
            "pose_minus_pcnose_ms": round(signed_delta_ms, 6),
            "abs_pose_minus_pcnose_ms": round(abs(signed_delta_ms), 6),
        }
    )
    if alignment_valid:
        row.update(pose.row)
    else:
        row.update({field: "" for field in POSE_FIELDS})
    return row


def create_video_writer(
    path: Path,
    image: np.ndarray,
    fps: float,
) -> cv2.VideoWriter:
    height, width = image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path}")
    return writer


def draw_live_view(
    pose: PoseSample,
    rows_written: int,
    latest_flag: object,
    latest_sync_ms: float | None,
    latest_alignment_valid: bool | None,
    max_sync_ms: float,
    alignment_stats: AlignmentStats,
    T_cube_tag: dict[int, np.ndarray],
    p_snout_cube_mm: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    output = pose.camera_frame.image.copy()
    pose_tracking.display.draw_tag_observations(output, pose.observations)

    if pose.T_camera_cube is not None:
        axis_pixels = pose_tracking.geometry.project_points(
            pose_tracking.geometry.cube_axis_points(),
            pose.T_camera_cube,
            camera_matrix,
        )
        pose_tracking.display.draw_axes(output, axis_pixels)
        plane_point = pose_tracking.geometry.snout_projection_on_id4(
            p_snout_cube_mm,
            T_cube_tag[4],
        )
        projected = pose_tracking.geometry.project_points(
            np.vstack([p_snout_cube_mm, plane_point]),
            pose.T_camera_cube,
            camera_matrix,
        )
        pose_tracking.display.draw_snout_geometry(output, projected[0], projected[1])

    desk_valid = bool(pose.row[f"tag_{pose_tracking.geometry.DESK_TAG_ID}_visible"])
    cube_valid = bool(pose.row["cube_pose_valid"])
    if latest_sync_ms is None:
        sync_text = "--"
    elif latest_alignment_valid:
        sync_text = f"{latest_sync_ms:+.1f} ms OK"
    else:
        sync_text = f"{latest_sync_ms:+.1f} ms REJECT > {max_sync_ms:g} ms"
    flag_text = "--" if latest_flag == "" else str(latest_flag)
    lines = [
        "LIVE: CYRANOSE READING + CALIBRATED POSE",
        "Space: finish | Close window: finish",
        f"Readings {rows_written} | Matched {alignment_stats.accepted} | "
        f"Rejected {alignment_stats.rejected}",
        f"Flag {flag_text} | Sync {sync_text}",
        f"Desk {'valid' if desk_valid else 'unavailable'} | Cube {'valid' if cube_valid else 'unavailable'}",
    ]
    pose_tracking.display.draw_text_lines(output, lines)
    return output


def show_guide(
    csv_path: Path,
    video_path: Path | None,
    port: str,
    baud: int,
    interval_s: float,
    max_sync_ms: float,
    probe_response: str,
    trial_id: str,
    trial_label: str,
) -> None:
    video_text = (
        f"Clean rectified RGB video: {video_path}"
        if video_path is not None
        else "Video recording is disabled."
    )
    print(
        "\nCYRANOSE READING + CALIBRATED POSE RECORDING\n\n"
        "The Cyranose is polled directly with RSD. Each valid SD reading is paired\n"
        "with the temporally nearest rectified camera frame. A match farther than\n"
        f"{max_sync_ms:g} ms is retained as a Cyranose row but its pose is left blank.\n"
        "The CSV stores both signed and absolute offsets for quality control.\n\n"
        "Tag, cube, and snout poses use the current calibration. Positions are in\n"
        "centimeters and Rodrigues rotations are in radians. Desk coordinates are\n"
        "available only while qualified desk tag ID 6 is visible.\n\n"
        f"Device: {probe_response}\n"
        f"Serial: {port} at {baud} baud | RSD interval: {interval_s:.3f} s\n"
        f"Trial: {trial_id or 'unassigned'} | Label: {trial_label or 'unlabeled'}\n"
        f"Recording immediately to: {csv_path}\n"
        f"{video_text}\n\n"
        "Press Space or close the window to finish and preserve recorded data.\n"
    )


def write_matches(
    writer: csv.DictWriter,
    matches: list[tuple[CyranoseReading, PoseSample]],
    max_sync_ms: float,
    alignment_stats: AlignmentStats,
) -> tuple[int, object, float | None, bool | None]:
    count = 0
    latest_flag: object = ""
    latest_sync_ms: float | None = None
    latest_alignment_valid: bool | None = None
    for reading, pose in matches:
        row = build_combined_row(reading, pose, max_sync_ms)
        writer.writerow(row)
        count += 1
        latest_flag = reading.decoded["flag"]
        latest_sync_ms = float(row["pose_minus_pcnose_ms"])
        latest_alignment_valid = alignment_stats.add(
            latest_sync_ms,
            reading.serial_roundtrip_ms,
        )
    return count, latest_flag, latest_sync_ms, latest_alignment_valid


def print_alignment_summary(summary: dict[str, Any]) -> None:
    absolute = summary["absolute_pose_minus_pcnose_ms"]
    serial = summary["serial_roundtrip_ms"]
    print(
        "Alignment QC: "
        f"{summary['matched_readings']}/{summary['total_readings']} matched "
        f"({summary['match_rate_percent']}%) within "
        f"{summary['max_sync_ms']:g} ms"
    )
    if summary["total_readings"]:
        print(
            "  Absolute pose/readout delta: "
            f"median {absolute['median']:.1f} ms, "
            f"p95 {absolute['p95']:.1f} ms, "
            f"p99 {absolute['p99']:.1f} ms, "
            f"max {absolute['maximum']:.1f} ms"
        )
        print(
            "  Serial request round trip: "
            f"median {serial['median']:.1f} ms, "
            f"p95 {serial['p95']:.1f} ms, "
            f"max {serial['maximum']:.1f} ms"
        )


def run_recording(
    port: str = DEFAULT_PORT,
    baud: int = DEFAULT_BAUD,
    interval_s: float = DEFAULT_INTERVAL_S,
    max_sync_ms: float = DEFAULT_MAX_SYNC_MS,
    save_video: bool = False,
    trial_id: str = "",
    trial_label: str = "",
    notes: str = "",
) -> None:
    if interval_s < 0:
        raise ValueError("--interval must be non-negative")
    if max_sync_ms <= 0:
        raise ValueError("--max-sync-ms must be positive")

    intrinsics, T_cube_tag, p_snout_cube_mm = pose_tracking.load_calibration()
    camera_matrix = np.asarray(intrinsics["rectified_K"], dtype=np.float64)
    output_dir, csv_path = session_paths(datetime.now())
    video_path = output_dir / VIDEO_NAME if save_video else None
    alignment_summary_path = output_dir / ALIGNMENT_SUMMARY_NAME
    session_metadata_path = output_dir / SESSION_METADATA_NAME

    reader = CyranoseReaderThread(port, baud, interval_s)
    reader.start()
    reader.ready.wait()
    if reader.error is not None:
        reader.join()
        raise RuntimeError(f"Cyranose initialization failed: {reader.error}") from reader.error
    if reader.probe_response is None:
        reader.request_stop()
        reader.join()
        raise RuntimeError("Cyranose initialization ended without an RVN response")

    video_writer: cv2.VideoWriter | None = None
    window_created = False
    reader_stopped = False
    pending: deque[CyranoseReading] = deque()
    previous_pose: PoseSample | None = None
    alignment_stats = AlignmentStats.create(max_sync_ms)

    try:
        with pose_tracking.RectifiedColorCamera(intrinsics) as camera:
            output_dir.mkdir(parents=True, exist_ok=False)
            session_metadata = {
                "session_id": output_dir.name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
                "trial_id": trial_id.strip(),
                "trial_label": trial_label.strip(),
                "notes": notes.strip(),
                "port": port,
                "baud": baud,
                "rsd_interval_s": interval_s,
                "max_sync_ms": max_sync_ms,
                "save_video": save_video,
            }
            with session_metadata_path.open("w", encoding="utf-8") as metadata_file:
                json.dump(session_metadata, metadata_file, indent=2)
                metadata_file.write("\n")
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            window_created = True
            show_guide(
                csv_path,
                video_path,
                port,
                baud,
                interval_s,
                max_sync_ms,
                reader.probe_response,
                trial_id,
                trial_label,
            )
            session_start_ns = time.perf_counter_ns()

            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
                writer.writeheader()
                csv_file.flush()
                reader.start_recording.set()

                rows_written = 0
                latest_flag: object = ""
                latest_sync_ms: float | None = None
                latest_alignment_valid: bool | None = None
                normal_stop = False

                try:
                    while True:
                        current_pose = capture_pose_sample(
                            camera,
                            session_start_ns,
                            camera_matrix,
                            T_cube_tag,
                            p_snout_cube_mm,
                        )
                        if save_video and video_writer is None:
                            video_writer = create_video_writer(
                                video_path,
                                current_pose.camera_frame.image,
                                float(intrinsics["fps"]),
                            )
                        if video_writer is not None:
                            video_writer.write(current_pose.camera_frame.image)

                        drain_readings(reader.readings, pending)
                        matches = match_ready_readings(
                            pending,
                            previous_pose,
                            current_pose,
                        )
                        count, flag, sync_ms, alignment_valid = write_matches(
                            writer,
                            matches,
                            max_sync_ms,
                            alignment_stats,
                        )
                        if count:
                            rows_written += count
                            latest_flag = flag
                            latest_sync_ms = sync_ms
                            latest_alignment_valid = alignment_valid
                            csv_file.flush()

                        live_view = draw_live_view(
                            current_pose,
                            rows_written,
                            latest_flag,
                            latest_sync_ms,
                            latest_alignment_valid,
                            max_sync_ms,
                            alignment_stats,
                            T_cube_tag,
                            p_snout_cube_mm,
                            camera_matrix,
                        )
                        cv2.imshow(WINDOW_NAME, live_view)
                        previous_pose = current_pose

                        if reader.error is not None:
                            if not pending or pending[-1].monotonic_ns <= current_pose.monotonic_ns:
                                raise RuntimeError(
                                    f"Cyranose recording failed: {reader.error}"
                                ) from reader.error

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord(" ") or pose_tracking.display.window_was_closed(
                            WINDOW_NAME
                        ):
                            normal_stop = True
                            break
                except KeyboardInterrupt:
                    normal_stop = True

                if normal_stop:
                    reader.request_stop()
                    reader.join()
                    reader_stopped = True
                    drain_readings(reader.readings, pending)
                    if pending:
                        final_pose = capture_pose_sample(
                            camera,
                            session_start_ns,
                            camera_matrix,
                            T_cube_tag,
                            p_snout_cube_mm,
                        )
                        if video_writer is not None:
                            video_writer.write(final_pose.camera_frame.image)
                        matches = match_ready_readings(
                            pending,
                            previous_pose,
                            final_pose,
                        )
                        count, _, _, _ = write_matches(
                            writer,
                            matches,
                            max_sync_ms,
                            alignment_stats,
                        )
                        rows_written += count
                        csv_file.flush()
                    summary = alignment_stats.as_dict()
                    with alignment_summary_path.open(
                        "w",
                        encoding="utf-8",
                    ) as summary_file:
                        json.dump(summary, summary_file, indent=2)
                        summary_file.write("\n")
                    print(f"Saved {rows_written} Cyranose readings to: {csv_path}")
                    print_alignment_summary(summary)
                    print(f"Saved alignment summary to: {alignment_summary_path}")
                    if video_path is not None:
                        print(f"Saved rectified RGB video to: {video_path}")
    finally:
        if not reader_stopped:
            reader.request_stop()
            reader.join()
        if video_writer is not None:
            video_writer.release()
        if window_created:
            cv2.destroyAllWindows()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record direct Cyranose 320 readings with temporally matched calibrated poses."
        )
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=f"Cyranose serial port, default {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Cyranose baud rate, default {DEFAULT_BAUD}.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help=f"Seconds between RSD requests, default {DEFAULT_INTERVAL_S}.",
    )
    parser.add_argument(
        "--max-sync-ms",
        type=float,
        default=DEFAULT_MAX_SYNC_MS,
        help=(
            "Maximum absolute Cyranose-to-pose time difference accepted as a "
            f"match, default {DEFAULT_MAX_SYNC_MS:g} ms."
        ),
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save clean rectified RGB frames as rectified_rgb.mp4.",
    )
    parser.add_argument(
        "--trial-id",
        default="",
        help="Optional independent trial identifier, for example mint_01.",
    )
    parser.add_argument(
        "--trial-label",
        default="",
        help="Optional class label such as blank, mint, or mint_line.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional short acquisition notes stored in session_metadata.json.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        run_recording(
            port=args.port,
            baud=args.baud,
            interval_s=args.interval,
            max_sync_ms=args.max_sync_ms,
            save_video=args.save_video,
            trial_id=args.trial_id,
            trial_label=args.trial_label,
            notes=args.notes,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

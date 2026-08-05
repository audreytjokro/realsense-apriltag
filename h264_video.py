"""Small FFmpeg-backed H.264 MP4 writer for OpenCV BGR frames."""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np


DEFAULT_CRF = 18
DEFAULT_PRESET = "veryfast"


class H264VideoWriter:
    """Write uint8 BGR frames to an H.264 MP4 through FFmpeg.

    OpenCV builds frequently expose H.264 decoders without an H.264 encoder.
    Feeding raw frames to the system FFmpeg makes the output codec explicit and
    consistent.  The destination is replaced only after FFmpeg exits cleanly.
    """

    def __init__(
        self,
        path: Path | str,
        fps: float,
        frame_size: tuple[int, int],
        *,
        crf: int = DEFAULT_CRF,
        preset: str = DEFAULT_PRESET,
        ffmpeg: Path | str | None = None,
    ) -> None:
        self.path = Path(path)
        self.fps = float(fps)
        self.width, self.height = (int(frame_size[0]), int(frame_size[1]))
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("H.264 video output must use an .mp4 extension")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Video FPS must be finite and positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Video dimensions must be positive")
        if self.width % 2 or self.height % 2:
            raise ValueError("yuv420p H.264 output requires even dimensions")
        if not 0 <= int(crf) <= 51:
            raise ValueError("H.264 CRF must be between 0 and 51")

        executable = str(ffmpeg) if ffmpeg is not None else shutil.which("ffmpeg")
        if not executable:
            raise RuntimeError(
                "FFmpeg is required for H.264 MP4 output; install ffmpeg and add it to PATH"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._temporary_path = self.path.with_name(
            f".{self.path.stem}.{uuid.uuid4().hex}.h264-writing.mp4"
        )
        command = [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            f"{self.fps:.12g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            str(preset),
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self._temporary_path),
        ]
        creationflags = (
            subprocess.CREATE_NO_WINDOW
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        self._released = False

    def isOpened(self) -> bool:
        return not self._released and self._process.poll() is None

    def write(self, frame: np.ndarray) -> None:
        if not self.isOpened() or self._process.stdin is None:
            raise RuntimeError("H.264 video writer is not open")
        image = np.asarray(frame)
        expected = (self.height, self.width, 3)
        if image.shape != expected or image.dtype != np.uint8:
            raise ValueError(
                f"Video frame must have shape {expected} and dtype uint8, "
                f"received {image.shape} and {image.dtype}"
            )
        try:
            self._process.stdin.write(np.ascontiguousarray(image).tobytes())
        except BrokenPipeError as error:
            message = self._read_stderr()
            raise RuntimeError(f"FFmpeg stopped while writing {self.path}: {message}") from error

    def _read_stderr(self) -> str:
        if self._process.stderr is None:
            return "no FFmpeg diagnostic was available"
        return self._process.stderr.read().decode("utf-8", errors="replace").strip()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            if self._process.stdin is not None and not self._process.stdin.closed:
                self._process.stdin.close()
            message = self._read_stderr()
            returncode = self._process.wait()
            if returncode != 0:
                raise RuntimeError(
                    f"FFmpeg failed to create {self.path} (exit {returncode}): {message}"
                )
            if not self._temporary_path.is_file() or self._temporary_path.stat().st_size == 0:
                raise RuntimeError(f"FFmpeg created no usable output for {self.path}")
            self._temporary_path.replace(self.path)
        except BaseException:
            self._temporary_path.unlink(missing_ok=True)
            raise
        finally:
            if self._process.stderr is not None:
                self._process.stderr.close()

    def __enter__(self) -> "H264VideoWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


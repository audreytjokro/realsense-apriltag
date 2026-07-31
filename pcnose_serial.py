from __future__ import annotations

import ctypes
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


ADC_COUNTS = 1 << 24
ADC_REF_VOLTS = 2.5
SENSOR_REF_KOHM = 2.21
HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


class PCNoseError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_hex_token(token: str) -> int:
    token = token.strip()
    if not HEX_RE.match(token):
        raise ValueError(f"not a hex token: {token!r}")
    return int(token, 16)


def active_sensor_indices(mask: int) -> list[int]:
    return [index + 1 for index in range(32) if mask & (1 << index)]


def adc_to_volts(value: int, adc_ref_volts: float = ADC_REF_VOLTS) -> float:
    return value * adc_ref_volts / ADC_COUNTS


def sensor_adc_to_kohm(
    value: int,
    sensor_ref_kohm: float = SENSOR_REF_KOHM,
) -> float:
    denominator = ADC_COUNTS - value
    if denominator <= 0:
        raise ValueError(f"sensor ADC value out of range: {value}")
    return sensor_ref_kohm * value / denominator


def format_float(value: float, places: int) -> str:
    return f"{value:.{places}f}"


@dataclass(frozen=True)
class SDFrame:
    raw: str
    field1: int
    field2: int
    sensor_mask: int
    sensors: list[int | None]
    misc: list[int]

    @property
    def active_sensors(self) -> list[int]:
        return active_sensor_indices(self.sensor_mask)

    @property
    def therm1_raw(self) -> int | None:
        return self.misc[0] if len(self.misc) >= 1 else None

    @property
    def battery_raw(self) -> int | None:
        return self.misc[1] if len(self.misc) >= 2 else None

    @property
    def flag(self) -> int | None:
        return self.misc[2] if len(self.misc) >= 3 else None

    @property
    def device_tick_deciseconds(self) -> int | None:
        return self.misc[3] if len(self.misc) >= 4 else None

    def as_decoded_row(
        self,
        host_time_utc: str,
        first_tick_deciseconds: int | None,
        sensor_ref_kohm: float = SENSOR_REF_KOHM,
        include_raw_frame: bool = True,
    ) -> dict[str, object]:
        tick = self.device_tick_deciseconds
        row: dict[str, object] = {
            "host_time_utc": host_time_utc,
            "elapsed_s": ""
            if tick is None or first_tick_deciseconds is None
            else format_float((tick - first_tick_deciseconds) / 10.0, 6),
            "device_time_s": "" if tick is None else format_float(tick / 10.0, 6),
            "therm1_volts": ""
            if self.therm1_raw is None
            else format_float(adc_to_volts(self.therm1_raw), 6),
            "battery_raw": "" if self.battery_raw is None else self.battery_raw,
            "flag": "" if self.flag is None else self.flag,
            "sd_field1_hex": f"{self.field1:02X}",
            "sd_field2_hex": f"{self.field2:02X}",
            "sensor_mask_hex": f"{self.sensor_mask:08X}",
            "active_sensor_count": len(self.active_sensors),
        }
        for index, value in enumerate(self.sensors, start=1):
            row[f"S{index}_kohm"] = (
                ""
                if value is None
                else format_float(sensor_adc_to_kohm(value, sensor_ref_kohm), 8)
            )
        if include_raw_frame:
            row["raw_frame"] = self.raw
        return row


def parse_sd_frame(line: str) -> SDFrame:
    tokens = line.strip().split()
    if not tokens or tokens[0].upper() != "SD":
        raise ValueError("frame does not start with SD")
    if len(tokens) < 4:
        raise ValueError("SD frame is too short")

    field1 = parse_hex_token(tokens[1])
    field2 = parse_hex_token(tokens[2])
    sensor_mask = parse_hex_token(tokens[3])
    payload = [parse_hex_token(token) for token in tokens[4:]]

    sensors: list[int | None] = [None] * 32
    active = active_sensor_indices(sensor_mask)
    if len(payload) >= 32:
        for index in range(32):
            sensors[index] = payload[index]
        misc = payload[32:]
    elif len(payload) >= len(active):
        for sensor_index, value in zip(active, payload[: len(active)]):
            sensors[sensor_index - 1] = value
        misc = payload[len(active) :]
    else:
        raise ValueError(
            f"SD payload has {len(payload)} values, "
            f"fewer than active sensor count {len(active)}"
        )

    return SDFrame(
        raw=line.strip(),
        field1=field1,
        field2=field2,
        sensor_mask=sensor_mask,
        sensors=sensors,
        misc=misc,
    )


def decoded_field_names(include_raw_frame: bool = True) -> list[str]:
    fields = [
        "host_time_utc",
        "elapsed_s",
        "device_time_s",
        "therm1_volts",
        "battery_raw",
        "flag",
        "sd_field1_hex",
        "sd_field2_hex",
        "sensor_mask_hex",
        "active_sensor_count",
    ]
    fields.extend(f"S{index}_kohm" for index in range(1, 33))
    if include_raw_frame:
        fields.append("raw_frame")
    return fields


if os.name == "nt":
    from ctypes import wintypes

    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    PURGE_RXCLEAR = 0x0008
    PURGE_TXCLEAR = 0x0004
    NOPARITY = 0
    ONESTOPBIT = 0
    DTR_CONTROL_ENABLE = 1
    RTS_CONTROL_ENABLE = 1

    class DCB(ctypes.Structure):
        _fields_ = [
            ("DCBlength", wintypes.DWORD),
            ("BaudRate", wintypes.DWORD),
            ("fBinary", wintypes.DWORD, 1),
            ("fParity", wintypes.DWORD, 1),
            ("fOutxCtsFlow", wintypes.DWORD, 1),
            ("fOutxDsrFlow", wintypes.DWORD, 1),
            ("fDtrControl", wintypes.DWORD, 2),
            ("fDsrSensitivity", wintypes.DWORD, 1),
            ("fTXContinueOnXoff", wintypes.DWORD, 1),
            ("fOutX", wintypes.DWORD, 1),
            ("fInX", wintypes.DWORD, 1),
            ("fErrorChar", wintypes.DWORD, 1),
            ("fNull", wintypes.DWORD, 1),
            ("fRtsControl", wintypes.DWORD, 2),
            ("fAbortOnError", wintypes.DWORD, 1),
            ("fDummy2", wintypes.DWORD, 17),
            ("wReserved", wintypes.WORD),
            ("XonLim", wintypes.WORD),
            ("XoffLim", wintypes.WORD),
            ("ByteSize", ctypes.c_ubyte),
            ("Parity", ctypes.c_ubyte),
            ("StopBits", ctypes.c_ubyte),
            ("XonChar", ctypes.c_char),
            ("XoffChar", ctypes.c_char),
            ("ErrorChar", ctypes.c_char),
            ("EofChar", ctypes.c_char),
            ("EvtChar", ctypes.c_char),
            ("wReserved1", wintypes.WORD),
        ]

    class COMMTIMEOUTS(ctypes.Structure):
        _fields_ = [
            ("ReadIntervalTimeout", wintypes.DWORD),
            ("ReadTotalTimeoutMultiplier", wintypes.DWORD),
            ("ReadTotalTimeoutConstant", wintypes.DWORD),
            ("WriteTotalTimeoutMultiplier", wintypes.DWORD),
            ("WriteTotalTimeoutConstant", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetCommState.argtypes = [wintypes.HANDLE, ctypes.POINTER(DCB)]
    kernel32.GetCommState.restype = wintypes.BOOL
    kernel32.SetCommState.argtypes = [wintypes.HANDLE, ctypes.POINTER(DCB)]
    kernel32.SetCommState.restype = wintypes.BOOL
    kernel32.SetCommTimeouts.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(COMMTIMEOUTS),
    ]
    kernel32.SetCommTimeouts.restype = wintypes.BOOL
    kernel32.SetupComm.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
    kernel32.SetupComm.restype = wintypes.BOOL
    kernel32.PurgeComm.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.PurgeComm.restype = wintypes.BOOL
    kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel32.ReadFile.restype = wintypes.BOOL
    kernel32.WriteFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPCVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel32.WriteFile.restype = wintypes.BOOL


def win_error(prefix: str) -> PCNoseError:
    return PCNoseError(f"{prefix}: {ctypes.WinError(ctypes.get_last_error())}")


def normalize_port_name(port: str) -> str:
    port = port.strip()
    if port.startswith("\\\\.\\"):
        return port
    return "\\\\.\\" + port


class WinSerial:
    def __init__(
        self,
        port: str,
        baud: int,
        read_timeout_ms: int = 120,
        write_timeout_ms: int = 1000,
    ) -> None:
        if os.name != "nt":
            raise PCNoseError("serial I/O requires Windows")
        self.port = port
        self.baud = baud
        self.handle = kernel32.CreateFileW(
            normalize_port_name(port),
            GENERIC_READ | GENERIC_WRITE,
            0,
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            None,
        )
        if self.handle == INVALID_HANDLE_VALUE:
            raise win_error(f"open {port}")
        try:
            self.configure(read_timeout_ms, write_timeout_ms)
        except Exception:
            self.close()
            raise

    def configure(self, read_timeout_ms: int, write_timeout_ms: int) -> None:
        if not kernel32.SetupComm(self.handle, 4096, 4096):
            raise win_error("SetupComm")

        dcb = DCB()
        dcb.DCBlength = ctypes.sizeof(DCB)
        if not kernel32.GetCommState(self.handle, ctypes.byref(dcb)):
            raise win_error("GetCommState")
        dcb.BaudRate = self.baud
        dcb.ByteSize = 8
        dcb.Parity = NOPARITY
        dcb.StopBits = ONESTOPBIT
        dcb.fBinary = 1
        dcb.fParity = 0
        dcb.fOutxCtsFlow = 0
        dcb.fOutxDsrFlow = 0
        dcb.fDtrControl = DTR_CONTROL_ENABLE
        dcb.fDsrSensitivity = 0
        dcb.fTXContinueOnXoff = 0
        dcb.fOutX = 0
        dcb.fInX = 0
        dcb.fErrorChar = 0
        dcb.fNull = 0
        dcb.fRtsControl = RTS_CONTROL_ENABLE
        dcb.fAbortOnError = 0
        if not kernel32.SetCommState(self.handle, ctypes.byref(dcb)):
            raise win_error("SetCommState")

        timeouts = COMMTIMEOUTS(
            ReadIntervalTimeout=50,
            ReadTotalTimeoutMultiplier=0,
            ReadTotalTimeoutConstant=read_timeout_ms,
            WriteTotalTimeoutMultiplier=0,
            WriteTotalTimeoutConstant=write_timeout_ms,
        )
        if not kernel32.SetCommTimeouts(self.handle, ctypes.byref(timeouts)):
            raise win_error("SetCommTimeouts")
        self.purge()

    def purge(self) -> None:
        if not kernel32.PurgeComm(self.handle, PURGE_RXCLEAR | PURGE_TXCLEAR):
            raise win_error("PurgeComm")

    def close(self) -> None:
        if getattr(self, "handle", None) not in (None, INVALID_HANDLE_VALUE):
            kernel32.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self) -> WinSerial:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def write_bytes(self, data: bytes) -> None:
        written = wintypes.DWORD()
        buffer = ctypes.create_string_buffer(data)
        ok = kernel32.WriteFile(
            self.handle,
            buffer,
            len(data),
            ctypes.byref(written),
            None,
        )
        if not ok:
            raise win_error("WriteFile")
        if written.value != len(data):
            raise PCNoseError(
                f"short serial write: {written.value}/{len(data)} bytes"
            )

    def read_bytes(self, size: int = 1) -> bytes:
        output = ctypes.create_string_buffer(size)
        count = wintypes.DWORD()
        ok = kernel32.ReadFile(
            self.handle,
            output,
            size,
            ctypes.byref(count),
            None,
        )
        if not ok:
            raise win_error("ReadFile")
        return output.raw[: count.value]

    def send_frame(self, frame: str) -> None:
        self.write_bytes(frame.strip().encode("ascii") + b"\r")

    def read_line(self, timeout_s: float = 2.0, max_bytes: int = 4096) -> str:
        deadline = time.monotonic() + timeout_s
        buffer = bytearray()
        while time.monotonic() < deadline:
            chunk = self.read_bytes(1)
            if not chunk:
                continue
            byte = chunk[0]
            if byte in (10, 13, 0x1A):
                if buffer:
                    return bytes(buffer).decode("ascii", errors="replace").strip()
                continue
            buffer.append(byte)
            if len(buffer) > max_bytes:
                raise PCNoseError(f"line exceeded {max_bytes} bytes")
        raise TimeoutError("serial line read timed out")


def read_responses(
    serial: WinSerial,
    timeout_s: float,
    prefixes: Iterable[str] | None = None,
) -> tuple[str | None, list[str]]:
    wanted = tuple(prefix.upper() for prefix in prefixes) if prefixes else ()
    seen: list[str] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = max(0.01, deadline - time.monotonic())
        try:
            line = serial.read_line(timeout_s=remaining)
        except TimeoutError:
            continue
        if not line:
            continue
        seen.append(line)
        head = line.split(maxsplit=1)[0].upper()
        if not wanted or head in wanted:
            return line, seen
    return None, seen


def request_frame(
    serial: WinSerial,
    command: str,
    prefixes: Iterable[str],
    timeout_s: float,
    attempts: int = 1,
) -> tuple[str, list[str]]:
    all_seen: list[str] = []
    for _ in range(attempts):
        serial.send_frame(command)
        line, seen = read_responses(serial, timeout_s=timeout_s, prefixes=prefixes)
        all_seen.extend(seen)
        if line is not None:
            return line, all_seen
    raise TimeoutError(f"no response to {command!r}; saw {all_seen!r}")

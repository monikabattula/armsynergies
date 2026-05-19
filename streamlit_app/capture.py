"""Threaded webcam capture: full 15 s session, START at 5 s, STOP at 11 s (wall clock)."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import custom_drawing_utils as cdu
from posehandDetector import posehandDetector
from video_recorder import VideoRecorder

from streamlit_app.task_registry import (
    MAX_RECORDED_FRAMES,
    OUTPUT_FPS,
    RECORD_COUNTDOWN_SEC,
    SESSION_CUE_START_SEC,
    SESSION_CUE_STOP_SEC,
    SESSION_DURATION_SEC,
)

_CAMERA_REOPEN_DELAY_SEC = 0.6
_CUE_BANNER_SEC = 1.0


def _setup_capture_quality(cap: cv2.VideoCapture) -> None:
    width = int(os.environ.get("CAMERA_WIDTH", "1280"))
    height = int(os.environ.get("CAMERA_HEIGHT", "720"))
    target_fps = float(os.environ.get("CAMERA_FPS", "30"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    if os.environ.get("CAMERA_AUTOFOCUS", "1").strip() in ("0", "false", "False"):
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)


def _draw_cue_banner(frame: np.ndarray, label: str, bgr: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    y0 = int(h * 0.32)
    y1 = int(h * 0.58)
    cv2.rectangle(overlay, (0, y0), (w, y1), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.52, frame, 0.48, 0, dst=frame)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = min(w / 280.0, h / 200.0)
    scale = max(1.35, min(scale, 3.2))
    thickness = max(3, int(round(scale)))
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    cx = (w - tw) // 2
    cy = (y0 + y1) // 2 + th // 2
    cv2.putText(frame, label, (cx, cy), font, scale, bgr, thickness, cv2.LINE_AA)


def _draw_countdown(frame: np.ndarray, seconds_left: int) -> None:
    label = str(max(1, seconds_left))
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = min(w / 120.0, h / 90.0)
    scale = max(2.5, min(scale, 6.0))
    thickness = max(4, int(round(scale * 1.2)))
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    cx = (w - tw) // 2
    cy = (h + th) // 2
    cv2.putText(frame, label, (cx, cy), font, scale, (80, 200, 255), thickness, cv2.LINE_AA)
    sub = "Get ready…"
    (sw, _), _ = cv2.getTextSize(sub, font, scale * 0.35, 2)
    cv2.putText(
        frame,
        sub,
        ((w - sw) // 2, int(h * 0.72)),
        font,
        scale * 0.35,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )


def _draw_session_timer(frame: np.ndarray, session_elapsed: float) -> None:
    left = max(0.0, SESSION_DURATION_SEC - session_elapsed)
    text = f"Session {session_elapsed:4.1f}s / {SESSION_DURATION_SEC:.0f}s  ({left:.1f}s left)"
    cv2.putText(
        frame,
        text,
        (8, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 165, 255),
        2,
    )


class LiveCapture:
    """
    One session: 15 s wall clock from Start.
    MP4 records the full 15 s; START cue at 5 s, STOP cue at 11 s.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._session_armed = threading.Event()
        self._session_active = False
        self._in_countdown = False
        self._recording = False
        self._error: Optional[str] = None
        self._saved_path: Optional[Path] = None
        self._record_output: Optional[Path] = None
        self._record_frame_count = 0
        self._session_t0: float = 0.0
        self._session_deadline: float = 0.0
        self._countdown_display: int = 0
        self._auto_stopped: bool = False

    @property
    def is_countdown(self) -> bool:
        return self._in_countdown

    @property
    def countdown_seconds_left(self) -> int:
        return self._countdown_display

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def record_frame_count(self) -> int:
        return self._record_frame_count

    @property
    def session_elapsed_sec(self) -> float:
        if self._session_t0 <= 0:
            return 0.0
        if not self.is_running and not self._session_active:
            return 0.0
        return min(
            SESSION_DURATION_SEC,
            max(0.0, time.monotonic() - self._session_t0),
        )

    @property
    def session_seconds_left(self) -> float:
        return max(0.0, SESSION_DURATION_SEC - self.session_elapsed_sec)

    @property
    def is_post_stop_cue(self) -> bool:
        """After STOP banner (11 s) until session end (15 s)."""
        e = self.session_elapsed_sec
        return self._recording and SESSION_CUE_STOP_SEC <= e < SESSION_DURATION_SEC

    @property
    def auto_stopped(self) -> bool:
        return self._auto_stopped

    def poll_completed_recording(self) -> Optional[Path]:
        if self.is_running:
            return None
        if self._saved_path is None:
            return None
        path = self._saved_path
        self._saved_path = None
        self._auto_stopped = False
        return path

    def get_latest_frame_rgb(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest is None:
                return None
            bgr = self._latest.copy()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def start_session(self, output_mp4: Path) -> Optional[str]:
        if self._session_active or self.is_running:
            return "A session is already in progress."
        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        self._error = None
        self._saved_path = None
        self._record_output = output_mp4
        self._record_frame_count = 0
        self._recording = False
        self._in_countdown = False
        self._auto_stopped = False
        self._session_t0 = 0.0
        self._session_deadline = 0.0
        self._session_active = True
        self._stop.clear()
        self._session_armed.set()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        for _ in range(150):
            if self._latest is not None or self._error:
                break
            if not self.is_running:
                break
            time.sleep(0.05)
        if self._error:
            self._session_active = False
            return self._error
        if not self.is_running:
            self._session_active = False
            return "Camera session failed to start."
        return None

    def stop(self) -> Optional[Path]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=25.0)
            self._thread = None
        self._stop.clear()
        self._session_armed.clear()
        self._session_active = False
        self._in_countdown = False
        self._recording = False
        self._session_t0 = 0.0
        path = self._saved_path
        self._saved_path = None
        self._auto_stopped = False
        with self._lock:
            self._latest = None
        return path

    def _begin_recorder(
        self,
        frame: np.ndarray,
        output_mp4: Path,
    ) -> VideoRecorder:
        os.environ["VIDEO_RECORD_DIR"] = str(output_mp4.parent)
        os.environ["RECORD_OUTPUT_FILE"] = str(output_mp4.resolve())
        # Fixed FPS → 15 s wall clock ≈ 15 s playback (see _sync_recorded_frames).
        os.environ["RECORD_FPS"] = str(OUTPUT_FPS)
        rec = VideoRecorder(str(_ROOT))
        h, w = frame.shape[:2]
        rec.start(time.monotonic(), w, h, fps=OUTPUT_FPS)
        self._recording = True
        self._record_frame_count = 0
        self._sync_recorded_frames(rec, frame, session_elapsed=0.0)
        return rec

    def _sync_recorded_frames(
        self,
        recorder: VideoRecorder,
        frame: np.ndarray,
        session_elapsed: float,
    ) -> None:
        """Write enough frames so file length matches session wall time at OUTPUT_FPS."""
        if not recorder.active:
            return
        target = int(session_elapsed * OUTPUT_FPS)
        target = min(max(0, target), MAX_RECORDED_FRAMES)
        while self._record_frame_count < target:
            recorder.write_frame(frame)
            self._record_frame_count += 1

    def _stop_recorder(
        self,
        recorder: Optional[VideoRecorder],
        output_mp4: Optional[Path],
    ) -> None:
        if recorder is None or not recorder.active:
            return
        raw = recorder.stop_and_save()
        if raw:
            out = output_mp4 or Path(raw)
            self._saved_path = out if out.is_file() else Path(raw)
        self._recording = False

    def _run_loop(self) -> None:
        cam_idx = int(os.environ.get("CAMERA_INDEX", "0"))
        if sys.platform == "win32":
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            self._error = "Camera not opened. Try CAMERA_INDEX=1 or allow camera access."
            self._session_active = False
            return

        _setup_capture_quality(cap)
        detector: Optional[posehandDetector] = None
        recorder: Optional[VideoRecorder] = None
        output_mp4: Optional[Path] = self._record_output
        read_failures = 0
        session_clock_started = False

        try:
            detector = posehandDetector(cdu.draw_landmarks)

            while not self._stop.is_set():
                now = time.monotonic()

                # Hard wall-clock end: always run full SESSION_DURATION_SEC unless user stops.
                if session_clock_started and now >= self._session_deadline:
                    self._auto_stopped = True
                    break

                ok, frame = cap.read()
                if not ok:
                    read_failures += 1
                    if read_failures > 60:
                        self._error = "Camera stopped delivering frames."
                        break
                    time.sleep(0.02)
                    continue
                read_failures = 0

                frame = cv2.flip(frame, 1)
                frame = detector.findWhole(frame)
                detector.findPosition(frame)

                if self._session_armed.is_set() and output_mp4 is not None:
                    self._session_armed.clear()
                    session_clock_started = True
                    self._session_t0 = now
                    self._session_deadline = now + SESSION_DURATION_SEC
                    self._in_countdown = True
                    try:
                        recorder = self._begin_recorder(frame, output_mp4)
                    except (OSError, RuntimeError, ValueError) as exc:
                        self._error = f"Could not start recording: {exc}"
                        break

                session_elapsed = (
                    (now - self._session_t0) if session_clock_started else 0.0
                )

                if session_clock_started and self._recording:
                    # 0–5 s: prep countdown (file is already recording).
                    if session_elapsed < SESSION_CUE_START_SEC:
                        self._in_countdown = True
                        left = SESSION_CUE_START_SEC - session_elapsed
                        self._countdown_display = max(1, int(left + 0.999))
                        _draw_countdown(frame, self._countdown_display)
                    else:
                        self._in_countdown = False

                    if (
                        SESSION_CUE_START_SEC
                        <= session_elapsed
                        < SESSION_CUE_START_SEC + _CUE_BANNER_SEC
                    ):
                        _draw_cue_banner(frame, "START", (80, 220, 80))
                    if (
                        SESSION_CUE_STOP_SEC
                        <= session_elapsed
                        < SESSION_CUE_STOP_SEC + _CUE_BANNER_SEC
                    ):
                        _draw_cue_banner(frame, "STOP", (60, 120, 255))

                    _draw_session_timer(frame, session_elapsed)
                    h, w = frame.shape[:2]
                    cv2.circle(frame, (w - 24, 24), 10, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        "REC",
                        (w - 120, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                    if recorder is not None and recorder.active:
                        self._sync_recorded_frames(recorder, frame, session_elapsed)

                with self._lock:
                    self._latest = frame
                time.sleep(0.001)

        except Exception as exc:
            self._error = str(exc)
        finally:
            fill_frame: Optional[np.ndarray] = None
            with self._lock:
                if self._latest is not None:
                    fill_frame = self._latest.copy()
            if (
                recorder is not None
                and recorder.active
                and session_clock_started
                and fill_frame is not None
            ):
                # Fill any remaining frames so the file is exactly ~15 s at OUTPUT_FPS.
                self._sync_recorded_frames(
                    recorder, fill_frame, SESSION_DURATION_SEC
                )
            if recorder is not None and recorder.active:
                self._stop_recorder(recorder, output_mp4)
            elif not self._recording:
                self._saved_path = None
            os.environ.pop("RECORD_OUTPUT_FILE", None)
            self._recording = False
            self._in_countdown = False
            self._session_active = False
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass
            cap.release()
            time.sleep(_CAMERA_REOPEN_DELAY_SEC)

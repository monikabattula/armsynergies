"""Landmark CSV export wrapper for recorded subject videos."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
_LANDMARK_EXPORT = _ROOT / "landmark_export"
if str(_LANDMARK_EXPORT) not in sys.path:
    sys.path.insert(0, str(_LANDMARK_EXPORT))

from holistic_pose_utils import (  # noqa: E402
    extract_pose_landmarks_from_video,
    write_landmarks_csv,
)

from streamlit_app.task_registry import (
    LANDMARK_END_SEC,
    LANDMARK_EXPORT_END_SEC,
    LANDMARK_EXPORT_START_SEC,
    LANDMARK_START_SEC,
    SESSION_CUE_STOP_SEC,
    SESSION_DURATION_SEC,
)


def video_duration_sec(video_path: str | Path) -> float:
    """Return container duration in seconds, or 0 if unknown."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps and fps > 0 and frames and frames > 0:
            return float(frames / fps)
    finally:
        cap.release()
    return 0.0


def export_window_for_video(
    duration_sec: float,
    start_sec: float = LANDMARK_START_SEC,
    end_sec: float = LANDMARK_END_SEC,
) -> tuple[float, float, str | None]:
    """
  Pick [start, end] for landmark export.
  Returns (start, end, warning_message).
    """
    if duration_sec <= 0:
        return start_sec, end_sec, "Could not read video duration."

    if duration_sec < start_sec:
        win_start, win_end = 0.0, duration_sec
        return (
            win_start,
            win_end,
            f"Clip is only {duration_sec:.1f}s long — exporting full clip "
            f"instead of {start_sec:g}–{end_sec:g}s.",
        )

    if duration_sec < end_sec:
        return (
            start_sec,
            duration_sec,
            f"Clip ends at {duration_sec:.1f}s — export window trimmed to "
            f"{start_sec:g}–{duration_sec:.1f}s.",
        )

    return start_sec, end_sec, None


def export_landmarks_for_recording(
    video_path: str | Path,
    csv_path: str | Path,
    *,
    start_sec: float = LANDMARK_EXPORT_START_SEC,
    end_sec: float = LANDMARK_EXPORT_END_SEC,
    pose_quality: str = "heavy",
) -> tuple[str | None, str | None]:
    """
    Run pose+hands extraction and write CSV.

    Returns (error_message, warning_message). error is None on success.
    """
    video_path = Path(video_path)
    csv_path = Path(csv_path)
    if not video_path.is_file():
        return f"Video not found: {video_path}", None

    duration = video_duration_sec(video_path)
    if duration > 0 and duration < 0.5:
        return (
            f"Video is too short ({duration:.2f}s, {int(max(1, duration * 30))} frame(s)). "
            f"Record the full {SESSION_DURATION_SEC:.0f} s session before stopping.",
            None,
        )

    win_start, win_end, window_warn = export_window_for_video(
        duration, start_sec, end_sec
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    min_cov = 90.0
    if duration > 0 and duration < 8.0:
        min_cov = 50.0

    res = extract_pose_landmarks_from_video(
        str(video_path),
        start_sec=win_start,
        end_sec=win_end,
        pose_quality=pose_quality,
        fill_missing_landmarks=True,
        min_xy_coverage_percent=min_cov,
    )
    if not res.ok:
        msg = res.error_message or "Landmark extraction failed."
        if duration > 0 and "No frames in export window" in msg:
            msg += (
                f" Video length is {duration:.1f}s — record at least "
                f"{SESSION_DURATION_SEC:.0f}s so the {start_sec:g}–{end_sec:g}s task window is included."
            )
        return msg, window_warn
    if res.dataframe is None or res.dataframe.empty:
        return "No landmark rows in export window.", window_warn

    write_landmarks_csv(res.dataframe, csv_path)
    return None, window_warn

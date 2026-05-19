"""
Backward-compatible name for video-only recording.

Older versions saved JSON; **that is removed**. ``LandmarkRecorder`` here is the same
as ``VideoRecorder`` — use ``.write_frame()`` after overlays, not ``.append()``.
"""

from __future__ import annotations

from video_recorder import (
    LandmarkRecorder,
    SessionRecorder,
    VideoRecorder,
)

__all__ = ["VideoRecorder", "SessionRecorder", "LandmarkRecorder"]

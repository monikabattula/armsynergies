"""
Live webcam recording as real **H.264 MP4** video files.

Preferred path: **pipe frames to the `ffmpeg` CLI** (libx264 + yuv420p). That produces
standard MP4 files that play in QuickTime, VLC, browsers, etc.

If `ffmpeg` is not in PATH, falls back to OpenCV ``VideoWriter`` (less reliable on macOS).

Environment:
  RECORD_BACKEND — ``auto`` (default): try ffmpeg first, then OpenCV; ``ffmpeg`` only;
                   ``opencv`` only.
  RECORD_FALLBACK_AVI, RECORD_FOURCC — see OpenCV fallback in _start_opencv_writer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Optional

import cv2
import numpy as np


def _align_dim(n: int, align: int = 4) -> int:
    n = int(n)
    n = max(align, n - (n % align))
    return n if n >= align else align


def _open_video_writer(
    path: str, fourcc: int, fps: float, frame_w: int, frame_h: int
) -> tuple[Optional[cv2.VideoWriter], str]:
    size = (frame_w, frame_h)
    if hasattr(cv2, "CAP_FFMPEG"):
        wri = cv2.VideoWriter(
            path, cv2.CAP_FFMPEG, fourcc, fps, size, True
        )
        if wri is not None and wri.isOpened():
            return wri, "CAP_FFMPEG"
        if wri is not None:
            wri.release()
    wri = cv2.VideoWriter(path, fourcc, fps, size, True)
    if wri is not None and wri.isOpened():
        return wri, "default"
    if wri is not None:
        wri.release()
    return None, ""


class VideoRecorder:
    """Record BGR frames to MP4: prefer ffmpeg (real video), else OpenCV."""

    def __init__(self, project_root: str) -> None:
        self.out_dir = os.environ.get("VIDEO_RECORD_DIR") or os.environ.get(
            "LANDMARK_RECORD_DIR", os.path.join(project_root, "recordings")
        )
        self.active = False
        self.recorded_count = 0
        self._writer: Optional[cv2.VideoWriter] = None
        self._ffmpeg: Optional[subprocess.Popen[bytes]] = None
        self._video_path: Optional[str] = None
        self._frame_size: tuple[int, int] = (0, 0)
        self._backend: str = "none"  # "none" | "ffmpeg" | "opencv"

    def start(self, _wall_time: float, width: int, height: int, fps: float = 30.0) -> None:
        self.active = False
        self.recorded_count = 0
        self._writer = None
        self._ffmpeg = None
        self._video_path = None
        self._backend = "none"

        fps = max(1.0, min(120.0, float(fps)))
        w = _align_dim(width, 4)
        h = _align_dim(height, 4)
        if w < 4 or h < 4:
            raise ValueError(f"Invalid frame size: {width}x{height}")
        self._frame_size = (w, h)

        os.makedirs(self.out_dir, exist_ok=True)
        out_abs = os.path.abspath(self.out_dir)
        ts = time.strftime("%Y%m%d_%H%M%S")
        custom_out = os.environ.get("RECORD_OUTPUT_FILE", "").strip()
        if custom_out:
            path_mp4 = custom_out
        else:
            path_mp4 = os.path.join(self.out_dir, f"pose_hands_{ts}.mp4")

        pref = os.environ.get("RECORD_BACKEND", "auto").strip().lower()
        if pref not in ("auto", "ffmpeg", "opencv"):
            pref = "auto"

        if os.path.isfile(path_mp4):
            try:
                os.remove(path_mp4)
            except OSError:
                pass

        # --- 1) ffmpeg pipe (real H.264 MP4) ---
        if pref in ("auto", "ffmpeg"):
            ff = shutil.which("ffmpeg")
            if ff is not None and self._try_start_ffmpeg(ff, path_mp4, w, h, fps, out_abs):
                self.active = True
                return
            if pref == "ffmpeg":
                raise RuntimeError(
                    "RECORD_BACKEND=ffmpeg but ffmpeg could not start. "
                    "Install ffmpeg (e.g. `brew install ffmpeg`) and ensure it is in PATH."
                )

        # --- 2) OpenCV fallback ---
        if pref in ("auto", "opencv"):
            if self._try_start_opencv(path_mp4, w, h, fps, out_abs, ts):
                self._backend = "opencv"
                self.active = True
                return

        raise RuntimeError(
            "Could not start recording (ffmpeg unavailable and OpenCV writer failed). "
            "Install: brew install ffmpeg   OR   reinstall OpenCV with FFmpeg support."
        )

    def _try_start_ffmpeg(
        self,
        ff_bin: str,
        path_mp4: str,
        w: int,
        h: int,
        fps: float,
        out_abs: str,
    ) -> bool:
        """Pipe raw BGR frames to libx264. Returns True if process started."""
        cmd = [
            ff_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{w}x{h}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            path_mp4,
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError:
            return False
        if proc.stdin is None:
            try:
                proc.kill()
            except OSError:
                pass
            return False
        self._ffmpeg = proc
        self._video_path = path_mp4
        self._backend = "ffmpeg"
        print(
            f"Recording (real video via ffmpeg → libx264 MP4):\n"
            f"  {path_mp4}\n"
            f"  size=({w}x{h}) fps={fps:.2f} out_dir={out_abs}",
            flush=True,
        )
        return True

    def _try_start_opencv(
        self, path_mp4: str, w: int, h: int, fps: float, out_abs: str, ts: str
    ) -> bool:
        trials: list[tuple[str, str]] = [
            ("mp4", "avc1"),
            ("mp4", "H264"),
            ("mp4", "mp4v"),
            ("mp4", "X264"),
        ]
        if os.environ.get("RECORD_FALLBACK_AVI", "").lower() in ("1", "true", "yes"):
            trials.extend([("avi", "MJPG"), ("avi", "XVID")])

        env_fourcc = os.environ.get("RECORD_FOURCC")
        env_ext = (os.environ.get("RECORD_EXT") or "mp4").lstrip(".")
        if env_fourcc and len(env_fourcc) == 4:
            trials.insert(0, (env_ext, env_fourcc))

        for ext, fc in trials:
            path = os.path.join(self.out_dir, f"pose_hands_{ts}.{ext}")
            if path != path_mp4 and os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            fourcc = cv2.VideoWriter_fourcc(*fc)
            wri, backend = _open_video_writer(path, fourcc, fps, w, h)
            if wri is not None:
                self._writer = wri
                self._video_path = path
                print(
                    f"Recording (OpenCV fallback — if playback fails, install ffmpeg):\n"
                    f"  {path}\n"
                    f"  size=({w}x{h}) fps={fps:.2f} fourcc={fc!r} backend={backend} out_dir={out_abs}",
                    flush=True,
                )
                return True
        return False

    def write_frame(self, frame_bgr: np.ndarray) -> None:
        if not self.active:
            return

        tw, th = self._frame_size
        fh, fw = frame_bgr.shape[:2]
        if fw != tw or fh != th:
            frame_bgr = cv2.resize(frame_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8, copy=False)
        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)

        if self._backend == "ffmpeg":
            if self._ffmpeg is None or self._ffmpeg.stdin is None:
                return
            try:
                self._ffmpeg.stdin.write(frame_bgr.tobytes())
            except (BrokenPipeError, OSError) as exc:
                print(f"ffmpeg pipe broken while recording: {exc}", file=sys.stderr)
                self.active = False
                return
            self.recorded_count += 1
            return

        if self._writer is None:
            return
        self._writer.write(frame_bgr)
        self.recorded_count += 1

    def stop_and_save(self) -> Optional[str]:
        if not self.active and self._ffmpeg is None and self._writer is None:
            return None

        self.active = False
        video_path = self._video_path

        if self._backend == "ffmpeg" and self._ffmpeg is not None:
            proc = self._ffmpeg
            self._ffmpeg = None
            err_msg = b""
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            # Do not use communicate() after closing stdin — Python 3.13 may try to
            # flush stdin and raise ValueError: flush of closed file.
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            err_msg = b""
            if proc.stderr is not None:
                err_msg = proc.stderr.read()
            rc = proc.returncode if proc.returncode is not None else -1
            self._backend = "none"

            if self.recorded_count == 0 or not video_path:
                if video_path and os.path.isfile(video_path):
                    try:
                        os.remove(video_path)
                    except OSError:
                        pass
                print(
                    "Recording stopped with 0 frames written.",
                    file=sys.stderr,
                )
                return None

            if rc != 0:
                print(
                    f"ffmpeg exited with code {rc}: {err_msg.decode(errors='replace')[:500]}",
                    file=sys.stderr,
                )
                if video_path and os.path.isfile(video_path):
                    try:
                        os.remove(video_path)
                    except OSError:
                        pass
                return None

            if video_path and os.path.isfile(video_path):
                sz = os.path.getsize(video_path)
                if sz < 256:
                    print(
                        f"Warning: output very small ({sz} bytes).",
                        file=sys.stderr,
                    )
            return video_path

        # OpenCV
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._backend = "none"

        if self.recorded_count == 0 or not video_path:
            if video_path and os.path.isfile(video_path):
                try:
                    os.remove(video_path)
                except OSError:
                    pass
            if self.recorded_count == 0:
                print(
                    "Recording stopped with 0 frames written (press R to start, "
                    "then R again after a few seconds).",
                    file=sys.stderr,
                )
            return None

        if video_path and os.path.isfile(video_path):
            sz = os.path.getsize(video_path)
            if sz < 256:
                print(
                    f"Warning: video file is very small ({sz} bytes). "
                    "Install ffmpeg for reliable MP4: brew install ffmpeg",
                    file=sys.stderr,
                )
        return video_path


SessionRecorder = VideoRecorder
LandmarkRecorder = VideoRecorder

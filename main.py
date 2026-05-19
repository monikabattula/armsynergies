
from __future__ import annotations

import os
import sys
import time
"""cv2 = webcam/video
cdu = custom landmark drawing"""

import cv2

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
"""posehandDetector= MediaPipe Detection class
VideoRecorder = saves mp4 recordings"""
import custom_drawing_utils as cdu
from posehandDetector import posehandDetector
from video_recorder import VideoRecorder


def _record_max_seconds() -> float:
    """0 = unlimited; otherwise auto-stop recording after this many seconds."""
    raw = os.environ.get("RECORD_MAX_SECONDS", "15").strip()
    if not raw or raw.lower() in ("0", "none", "off"):
        return 0.0
    return max(0.0, float(raw))


def _record_cue_seconds(name: str, default: float) -> float:
    """0 = disabled. Used for on-screen / console reminders during recording."""
    raw = os.environ.get(name, str(int(default)) if default == int(default) else str(default)).strip()
    if not raw or raw.lower() in ("0", "none", "off"):
        return 0.0
    return max(0.0, float(raw))


def _draw_recording_cue_banner(
    frame,
    label: str,
    bgr: tuple[int, int, int],
) -> None:
    """Large centered banner (burned into recorded frames when active)."""
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


def _writer_fps(cap: cv2.VideoCapture, smoothed: float) -> float:
    env = os.environ.get("RECORD_FPS")
    if env:
        return max(1.0, min(120.0, float(env)))
    # Prefer measured runtime FPS (already includes detector cost) so video duration
    # matches wall-clock recording time.
    if smoothed > 1.0:
        return float(min(60.0, max(8.0, smoothed)))
    cf = cap.get(cv2.CAP_PROP_FPS)
    if cf and cf > 1.0:
        return float(min(60.0, max(8.0, cf)))
    return 15.0


def _setup_capture_quality(cap: cv2.VideoCapture) -> None:
    """Best-effort camera settings for clearer landmark extraction input."""
    width = int(os.environ.get("CAMERA_WIDTH", "1280"))
    height = int(os.environ.get("CAMERA_HEIGHT", "720"))
    target_fps = float(os.environ.get("CAMERA_FPS", "30"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    # Optional: disable autofocus jitter if user requests fixed focus.
    if os.environ.get("CAMERA_AUTOFOCUS", "1").strip() in ("0", "false", "False"):
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)

"""opens webcam
default camera=0
can override using environment variable"""
def main() -> None:
    cam = int(os.environ.get("CAMERA_INDEX", "0"))
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Camera not opened — try CAMERA_INDEX=1, allow camera for PyCharm/Terminal.")
        return
    _setup_capture_quality(cap)

    detector = posehandDetector(cdu.draw_landmarks)
    # Video file only — see video_recorder.py (no JSON / landmark export)
    rec = VideoRecorder(_ROOT)
    prev = 0.0
    fps_smooth = 30.0
    rec_start_time: float | None = None
    rec_max_sec = _record_max_seconds()
    cue_start_sec = _record_cue_seconds("RECORD_CUE_START_SEC", 5.0)
    cue_stop_sec = _record_cue_seconds("RECORD_CUE_STOP_SEC", 12.0)
    cue_start_announced = False
    cue_stop_announced = False
    cue_banner_seconds = 1.0  # how long START / STOP stay on screen

    print(
        "Window open.\n"
        "  R — start/stop VIDEO recording (.mp4 in recordings/)\n"
        "  ESC — quit (finalizes active recording)\n"
        + (
            f"  Max clip length: {rec_max_sec:.0f} s (set RECORD_MAX_SECONDS=0 for no limit).\n"
            if rec_max_sec > 0
            else "  Max clip length: unlimited (RECORD_MAX_SECONDS=0).\n"
        )
        + (
            f"  Session cues (full clip still records): START ~{cue_start_sec:.0f}s, "
            f"STOP ~{cue_stop_sec:.0f}s (set RECORD_CUE_*_SEC=0 to disable).\n"
            if (cue_start_sec > 0 or cue_stop_sec > 0)
            else ""
        )
        + "  Stand about 1.5–2.5 m from the camera for best full-body + hands in frame.\n"
        "Click the video window first so key presses work. "
        "Prefer ESC over Ctrl+C; interrupt can show a long traceback."
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame = detector.findWhole(frame)
            pose_pts, left_pts, right_pts = detector.findPosition(frame)

            now = time.monotonic()
            if prev:
                inst = 1.0 / (now - prev) if (now - prev) > 0 else 0.0
                fps_smooth = fps_smooth * 0.9 + inst * 0.1
            prev = now

            status = (
                f"FPS:{int(fps_smooth)} "
                f"pose pts:{len(pose_pts)} Lhand:{len(left_pts)} Rhand:{len(right_pts)}"
            )
            if rec.active:
                status = f"REC {rec.recorded_count} frames | " + status

            cv2.putText(
                frame,
                status,
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255) if rec.active else (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "R record video | ESC quit",
                (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                frame,
                "Distance: stand 1.5-2.5 m from camera (full body + hands)",
                (8, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (180, 220, 255),
                1,
            )
            if rec.active and rec_start_time is not None and rec_max_sec > 0:
                elapsed = max(0.0, now - rec_start_time)
                left = max(0.0, rec_max_sec - elapsed)
                cv2.putText(
                    frame,
                    f"Time {elapsed:4.1f}s / {rec_max_sec:.0f}s  ({left:.1f}s left)",
                    (8, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2,
                )
                # Full-length recording continues; cues only remind when to begin / end the action.
                cap_ok_start = rec_max_sec <= 0 or rec_max_sec > cue_start_sec
                cap_ok_stop = rec_max_sec <= 0 or rec_max_sec > cue_stop_sec
                if (
                    cue_start_sec > 0
                    and cap_ok_start
                    and cue_start_sec <= elapsed < cue_start_sec + cue_banner_seconds
                ):
                    _draw_recording_cue_banner(frame, "START", (80, 220, 80))
                if (
                    cue_stop_sec > 0
                    and cap_ok_stop
                    and cue_stop_sec <= elapsed < cue_stop_sec + cue_banner_seconds
                ):
                    _draw_recording_cue_banner(frame, "STOP", (60, 120, 255))

                if cue_start_sec > 0 and cap_ok_start and elapsed >= cue_start_sec:
                    if not cue_start_announced:
                        cue_start_announced = True
                        print(
                            "\a>>> START — begin your movement / pose (recording continues).\n",
                            flush=True,
                        )
                if cue_stop_sec > 0 and cap_ok_stop and elapsed >= cue_stop_sec:
                    if not cue_stop_announced:
                        cue_stop_announced = True
                        tail = (
                            f"{rec_max_sec:.0f}s"
                            if rec_max_sec > 0
                            else "you press R / quit"
                        )
                        print(
                            f"\a>>> STOP — hold still or finish (recording continues to {tail}).\n",
                            flush=True,
                        )
            if rec.active:
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

            # Append this frame to the open video file (live recording)
            if rec.active:
                rec.write_frame(frame)

            # Auto-stop at max duration (default 15 s)
            if (
                rec.active
                and rec_start_time is not None
                and rec_max_sec > 0
                and (now - rec_start_time) >= rec_max_sec
            ):
                path = rec.stop_and_save()
                rec_start_time = None
                if path:
                    print(f"Saved video (max {rec_max_sec:.0f} s): {path}")
                else:
                    print("Recording stopped at time limit (no frames captured).")

            cv2.imshow("pose + hands", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("r"), ord("R")):
                if rec.active:
                    path = rec.stop_and_save()
                    rec_start_time = None
                    if path:
                        print(f"Saved video: {path}")
                    else:
                        print("Recording stopped (no frames captured).")
                else:
                    h, w = frame.shape[:2]
                    wf = _writer_fps(cap, fps_smooth)
                    try:
                        rec.start(now, w, h, fps=wf)
                    except (OSError, RuntimeError, ValueError) as exc:
                        print(f"Could not start recording: {exc}", file=sys.stderr)
                        continue
                    rec_start_time = time.monotonic()
                    cue_start_announced = False
                    cue_stop_announced = False
                    hint = (
                        f" — auto-stops at {rec_max_sec:.0f} s"
                        if rec_max_sec > 0
                        else " — no time limit"
                    )
                    print(
                        f"Recording video @ {wf:.1f} FPS{hint} — press R again to stop early."
                    )

    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")
    finally:
        if rec.active:
            path = rec.stop_and_save()
            rec_start_time = None
            if path:
                print(f"Saved video on exit: {path}")
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

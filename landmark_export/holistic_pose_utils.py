"""
Full-body pose (33 pts) + both hands (21+21) landmark extraction from video.

Uses MediaPipe **Tasks** `PoseLandmarker` + `HandLandmarker` in VIDEO mode.
Default export: **all 33 BlazePose landmarks** + hands for maximum coverage.

Face landmarks are not exported. Legacy `mediapipe.solutions` is not used.
"""

from __future__ import annotations

import math
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp

    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        PoseLandmarker,
        PoseLandmarkerOptions,
    )
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode as RunningMode,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mediapipe (tasks API) is required. Install: pip install mediapipe"
    ) from exc

COLUMN_ORDER = (
    "frame_id",
    "timestamp_sec",
    "landmark_type",
    "landmark_id",
    "x",
    "y",
    "z",
    "visibility",
)

LANDMARK_TYPE_POSE = "pose"
LANDMARK_TYPE_LEFT = "left_hand"
LANDMARK_TYPE_RIGHT = "right_hand"

HAND_NUM_LANDMARKS = 21

# Subset-only mode: 8 upper-body BlazePose indices (shoulders / arms / hips).
POSE_LANDMARK_INDICES_UPPER: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 23, 24)
# Backward-compatible name (older scripts).
POSE_LANDMARK_INDICES: tuple[int, ...] = POSE_LANDMARK_INDICES_UPPER

FULL_BODY_POSE_NUM = 33


def pose_indices_for_scope(scope: str) -> tuple[int, ...]:
    """``full`` = all 33 BlazePose points; ``upper`` = 8-point upper-body subset."""
    s = scope.lower().strip()
    if s == "full":
        return tuple(range(FULL_BODY_POSE_NUM))
    if s in ("upper", "upper_body"):
        return POSE_LANDMARK_INDICES_UPPER
    raise ValueError(f"pose_scope must be 'full' or 'upper', got {scope!r}")


def landmark_rows_per_frame(pose_scope: str) -> int:
    return len(pose_indices_for_scope(pose_scope)) + HAND_NUM_LANDMARKS * 2


# Default export is full body: 33 + 21 + 21 = 75 rows per frame.
ROWS_PER_FRAME: int = landmark_rows_per_frame("full")

# (local_filename, download_url). MEDIAPIPE_POSE_MODEL_PATH overrides any variant.
POSE_MODEL_VARIANTS: dict[str, tuple[str, str]] = {
    "lite": (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    ),
    "full": (
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    ),
    "heavy": (
        "pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    ),
}
_HAND_MODEL = (
    "hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task",
)

ProgressCallback = Callable[[int, int | None], None]


@dataclass(frozen=True)
class ProcessingResult:
    """Outcome of holistic landmark extraction."""

    dataframe: pd.DataFrame
    frames_processed: int
    ok: bool
    error_message: str | None = None


def _resize_frame_bgr(
    frame: np.ndarray, target_width: int | None
) -> np.ndarray:
    """Resize frame if wider than target_width; preserves aspect ratio."""
    if not target_width or target_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    new_w = target_width
    new_h = max(1, int(round(h * (target_width / w))))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _optional_luma_clahe(bgr: np.ndarray) -> np.ndarray:
    """Mild local contrast on luminance (helps dark/noisy phone or webcam clips)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l2 = clahe.apply(l_ch)
    merged = cv2.merge((l2, a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _mild_unsharp_bgr(bgr: np.ndarray, amount: float) -> np.ndarray:
    """Light unsharp mask on BGR for clearer edges (helps pose / finger tips)."""
    a = float(amount)
    if a <= 0.0:
        return bgr
    k = min(0.85, max(0.0, a))
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.05)
    return cv2.addWeighted(bgr, 1.0 + k, blurred, -k, 0)


def prepare_frame_for_inference(
    frame_bgr: np.ndarray,
    *,
    max_width: int | None = 1280,
    min_width: int | None = 960,
    enhance_luma: bool = False,
    unsharp_amount: float = 0.0,
) -> np.ndarray:
    """
    Spatial prep for clearer landmarks:

    1. **Upscale** narrow frames (e.g. 640×480 phone/webcam) so the model sees enough
       detail (uses cubic interpolation).
    2. **Downscale** only if wider than ``max_width`` (area interpolation).
    3. Optional **CLAHE** on luminance for dark recordings.
    4. Optional **unsharp** (``unsharp_amount`` > 0) after CLAHE for crisper limbs / hands.

    Landmarks are still normalized to this processed frame; match the same pipeline in verify.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    out = frame_bgr
    h, w = out.shape[:2]

    if min_width and min_width > 0 and w < min_width:
        scale = min_width / float(w)
        new_w = min_width
        new_h = max(1, int(round(h * scale)))
        out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if max_width and max_width > 0:
        out = _resize_frame_bgr(out, max_width)

    if enhance_luma:
        out = _optional_luma_clahe(out)

    if unsharp_amount and float(unsharp_amount) > 0.0:
        out = _mild_unsharp_bgr(out, float(unsharp_amount))

    return out


def _frame_total_hint(cap: cv2.VideoCapture) -> int | None:
    """Best-effort frame count from container; None if unknown or unreliable."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return total if total > 0 else None


def _model_cache_dir() -> Path:
    if os.name == "nt":
        base = Path(
            os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        )
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    d = base / "pose_landmark_extractor" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_task_file(
    filename: str,
    url: str,
    env_keys: tuple[str, ...],
) -> tuple[str | None, str | None]:
    """Return (local_path, error_message). Downloads the .task bundle if missing."""
    for key in env_keys:
        v = os.environ.get(key)
        if v:
            p = Path(v).expanduser().resolve()
            if p.is_file() and p.stat().st_size > 0:
                return str(p), None
            return None, f"Model path from {key} is missing or empty: {p}"

    dest = _model_cache_dir() / filename
    if dest.is_file() and dest.stat().st_size > 0:
        return str(dest), None

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        try:
            try:
                import certifi

                ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                ssl_ctx = ssl.create_default_context()
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=120, context=ssl_ctx) as resp:  # noqa: S310
                data = resp.read()
        except urllib.error.URLError as exc:
            return None, (
                f"Failed to download {filename}: {exc}. "
                "Try: pip install certifi, or set MEDIAPIPE_POSE_MODEL_PATH / "
                "MEDIAPIPE_HAND_MODEL_PATH to local .task files."
            )
        tmp.write_bytes(data)
        tmp.replace(dest)
    except OSError as exc:
        return None, f"Could not save model to {dest}: {exc}"
    finally:
        if tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass

    if not dest.is_file() or dest.stat().st_size == 0:
        return None, f"Model file is missing or empty: {dest}"
    return str(dest), None


def _ensure_pose_and_hand_models(
    pose_quality: str = "lite",
) -> tuple[str | None, str | None, str | None]:
    """Return (pose_model_path, hand_model_path, error_message)."""
    key = pose_quality.lower().strip()
    if key not in POSE_MODEL_VARIANTS:
        return None, None, (
            f"Unknown pose_quality {pose_quality!r}. Use one of: "
            f"{', '.join(sorted(POSE_MODEL_VARIANTS))}"
        )
    pfn, purl = POSE_MODEL_VARIANTS[key]
    hfn, hurl = _HAND_MODEL
    pp, err = _download_task_file(pfn, purl, ("MEDIAPIPE_POSE_MODEL_PATH",))
    if err or not pp:
        return None, None, err
    hp, err2 = _download_task_file(hfn, hurl, ("MEDIAPIPE_HAND_MODEL_PATH",))
    if err2 or not hp:
        return None, None, err2
    return pp, hp, None


def _normalize_xy_pixels(lm, frame_w: int, frame_h: int) -> tuple[float, float]:
    """
    MediaPipe NormalizedLandmarks are in [0,1]; convert via pixels to x/w, y/h
    (identical to lm.x, lm.y) for explicit normalization contract.
    """
    x_px = float(lm.x) * frame_w
    y_px = float(lm.y) * frame_h
    return x_px / float(frame_w), y_px / float(frame_h)


def _pose_vector(
    lm,
    frame_w: int,
    frame_h: int,
    nan: float,
) -> np.ndarray:
    """Four values [x, y, z, visibility] normalized / model units; NaNs if missing."""
    if lm is None or lm.x is None or lm.y is None:
        return np.array([nan, nan, nan, nan], dtype=np.float64)
    x_n, y_n = _normalize_xy_pixels(lm, frame_w, frame_h)
    z_v = float(lm.z) if lm.z is not None else nan
    vis = float(lm.visibility) if lm.visibility is not None else nan
    return np.array([x_n, y_n, z_v, vis], dtype=np.float64)


def _hand_matrix_from_landmarks(
    landmarks: Sequence,
    frame_w: int,
    frame_h: int,
    nan: float,
) -> np.ndarray:
    m = np.full((HAND_NUM_LANDMARKS, 4), nan, dtype=np.float64)
    n = min(HAND_NUM_LANDMARKS, len(landmarks))
    for hid in range(n):
        lm = landmarks[hid]
        if lm is not None and lm.x is not None and lm.y is not None:
            x_n, y_n = _normalize_xy_pixels(lm, frame_w, frame_h)
            m[hid, 0] = x_n
            m[hid, 1] = y_n
            m[hid, 2] = float(lm.z) if lm.z is not None else nan
            if lm.visibility is not None:
                m[hid, 3] = float(lm.visibility)
            else:
                m[hid, 3] = 1.0
    return m


def _smooth_landmark_block(
    raw: np.ndarray,
    state: np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    Exponential smoothing: ``state = lam * raw + (1 - lam) * state`` when both frames valid.

    ``lam`` is the weight on the **new** observation (0 = freeze previous, 1 = no smoothing).
    Missing (NaN x/y) clears that slot and passes NaNs through.
    """
    out = np.empty_like(raw)
    for i in range(raw.shape[0]):
        r = raw[i]
        if np.isnan(r[0]) or np.isnan(r[1]):
            state[i] = np.nan
            out[i] = r
            continue
        if lam <= 0.0 or lam >= 1.0:
            state[i] = r.copy()
            out[i] = r.copy()
            continue
        if np.isnan(state[i, 0]):
            state[i] = r.copy()
            out[i] = r.copy()
        else:
            blended = lam * r + (1.0 - lam) * state[i]
            state[i] = blended
            out[i] = blended.copy()
    return out


def _row_pose_from_vector(
    frame_id: int,
    timestamp_sec: float,
    landmark_id: int,
    vals: np.ndarray,
    nan: float,
) -> tuple[int, float, str, int, float, float, float, float]:
    if np.isnan(vals[0]) or np.isnan(vals[1]):
        return (
            frame_id,
            timestamp_sec,
            LANDMARK_TYPE_POSE,
            landmark_id,
            nan,
            nan,
            nan,
            nan,
        )
    z_v = float(vals[2]) if not np.isnan(vals[2]) else nan
    vis_v = float(vals[3]) if not np.isnan(vals[3]) else nan
    return (
        frame_id,
        timestamp_sec,
        LANDMARK_TYPE_POSE,
        landmark_id,
        float(vals[0]),
        float(vals[1]),
        z_v,
        vis_v,
    )


def _rows_hand_from_matrix(
    frame_id: int,
    timestamp_sec: float,
    m: np.ndarray,
    landmark_type: str,
    nan: float,
) -> list[tuple[int, float, str, int, float, float, float, float]]:
    """Build 21 CSV rows from a (21,4) x,y,z,vis matrix."""
    rows: list[tuple[int, float, str, int, float, float, float, float]] = []
    for hid in range(HAND_NUM_LANDMARKS):
        vals = m[hid]
        if np.isnan(vals[0]) or np.isnan(vals[1]):
            rows.append(
                (
                    frame_id,
                    timestamp_sec,
                    landmark_type,
                    hid,
                    nan,
                    nan,
                    nan,
                    nan,
                )
            )
        else:
            vis = float(vals[3]) if not np.isnan(vals[3]) else 1.0
            z_v = float(vals[2]) if not np.isnan(vals[2]) else nan
            rows.append(
                (
                    frame_id,
                    timestamp_sec,
                    landmark_type,
                    hid,
                    float(vals[0]),
                    float(vals[1]),
                    z_v,
                    vis,
                )
            )
    return rows


def _row_pose(
    frame_id: int,
    timestamp_sec: float,
    landmark_id: int,
    lm,
    frame_w: int,
    frame_h: int,
    nan: float,
) -> tuple[int, float, str, int, float, float, float, float]:
    if lm is None or lm.x is None or lm.y is None:
        return (
            frame_id,
            timestamp_sec,
            LANDMARK_TYPE_POSE,
            landmark_id,
            nan,
            nan,
            nan,
            nan,
        )
    x_n, y_n = _normalize_xy_pixels(lm, frame_w, frame_h)
    z_v = float(lm.z) if lm.z is not None else nan
    vis = float(lm.visibility) if lm.visibility is not None else nan
    return (
        frame_id,
        timestamp_sec,
        LANDMARK_TYPE_POSE,
        landmark_id,
        x_n,
        y_n,
        z_v,
        vis,
    )


def _rows_hand(
    frame_id: int,
    timestamp_sec: float,
    landmarks: Sequence,
    landmark_type: str,
    frame_w: int,
    frame_h: int,
    nan: float,
) -> list[tuple[int, float, str, int, float, float, float, float]]:
    """
    21 rows.

    MediaPipe Holistic hand landmarks typically do not populate ``has_visibility`` in
    the native API, so ``lm.visibility`` is almost always ``None``. The branch below
    then uses 1.0 per project CSV rules (not a model-supplied visibility score).
    """
    out: list[tuple[int, float, str, int, float, float, float, float]] = []
    for hid in range(HAND_NUM_LANDMARKS):
        if hid < len(landmarks):
            lm = landmarks[hid]
            if lm is not None and lm.x is not None and lm.y is not None:
                x_n, y_n = _normalize_xy_pixels(lm, frame_w, frame_h)
                z_v = float(lm.z) if lm.z is not None else nan
                # Hand task rarely sets visibility; default 1.0 is a placeholder only.
                if lm.visibility is not None:
                    vis = float(lm.visibility)
                else:
                    vis = 1.0
                out.append(
                    (
                        frame_id,
                        timestamp_sec,
                        landmark_type,
                        hid,
                        x_n,
                        y_n,
                        z_v,
                        vis,
                    )
                )
            else:
                out.append(
                    (
                        frame_id,
                        timestamp_sec,
                        landmark_type,
                        hid,
                        nan,
                        nan,
                        nan,
                        nan,
                    )
                )
        else:
            out.append(
                (
                    frame_id,
                    timestamp_sec,
                    landmark_type,
                    hid,
                    nan,
                    nan,
                    nan,
                    nan,
                )
            )
    return out


def _xy_coverage_percent(df: pd.DataFrame) -> float:
    """Percent of rows with both x and y present (non-NaN)."""
    if df.empty:
        return 0.0
    ok_xy = df["x"].notna() & df["y"].notna()
    return 100.0 * float(ok_xy.sum()) / float(len(df))


def _count_valid_xy_rows(m: np.ndarray) -> int:
    """Count rows where both x and y are present."""
    if m.size == 0:
        return 0
    return int(np.sum(~np.isnan(m[:, 0]) & ~np.isnan(m[:, 1])))


def _fill_missing_landmark_gaps(
    df: pd.DataFrame,
    max_gap: int = 2,
) -> pd.DataFrame:
    """
    Fill temporal gaps per (landmark_type, landmark_id) across frame_id.

    Uses linear interpolation, then forward/backward fill at edges so short
    visibility dropouts do not become missing CSV points.
    """
    if df.empty:
        return df
    out = df.copy()
    out = out.sort_values(["landmark_type", "landmark_id", "frame_id"], kind="mergesort")
    cols = ("x", "y", "z", "visibility")
    for _, idx in out.groupby(["landmark_type", "landmark_id"], sort=False).groups.items():
        block = out.loc[idx, list(cols)].copy()
        for c in cols:
            s = pd.to_numeric(block[c], errors="coerce").astype(np.float64)
            if s.notna().sum() == 0:
                continue
            s = s.interpolate(
                method="linear",
                limit=max(0, int(max_gap)),
                limit_direction="both",
            )
            s = s.ffill().bfill()
            block[c] = s
        out.loc[idx, list(cols)] = block
    return out


def _assign_left_right_hand_landmarks(hr) -> tuple[list, list]:
    """
    Map MediaPipe ``HandLandmarker`` output to (left_pts, right_pts), each a list of
    21 landmarks (may be empty if no hand).

    Uses handedness when reliable; if two hands share the same label or labels are
    missing, assigns the second / leftover hand by image-x so both sides are filled
    whenever two distinct hands are returned.
    """
    if not hr.hand_landmarks:
        return [], []

    entries: list[dict[str, object]] = []
    for i, pts in enumerate(hr.hand_landmarks):
        if not pts:
            continue
        p_list = list(pts)
        cats = hr.handedness[i] if (hr.handedness and i < len(hr.handedness)) else []
        name = (cats[0].category_name if cats else "") or ""
        score = (
            float(cats[0].score)
            if (cats and getattr(cats[0], "score", None) is not None)
            else 0.0
        )
        cx = float(np.mean([float(p.x) for p in p_list]))
        entries.append({"idx": i, "pts": p_list, "name": name, "score": score, "cx": cx})

    if not entries:
        return [], []

    left_tagged = [e for e in entries if e["name"] == "Left"]
    right_tagged = [e for e in entries if e["name"] == "Right"]

    def _best_pick(pool: list[dict[str, object]]) -> tuple[list | None, int | None]:
        if not pool:
            return None, None
        best = max(pool, key=lambda e: float(e["score"]))
        return best["pts"], int(best["idx"])

    left_pts, li = _best_pick(left_tagged)
    right_pts, ri = _best_pick(right_tagged)
    used: set[int] = {x for x in (li, ri) if x is not None}

    remainder = [e for e in entries if int(e["idx"]) not in used]
    remainder.sort(key=lambda e: float(e["cx"]))

    if left_pts is None and remainder:
        left_pts = remainder.pop(0)["pts"]
    if right_pts is None and remainder:
        right_pts = remainder.pop(-1)["pts"]
    if left_pts is None and remainder:
        left_pts = remainder.pop(0)["pts"]
    if right_pts is None and remainder:
        right_pts = remainder.pop(-1)["pts"]

    return list(left_pts or []), list(right_pts or [])


def extract_pose_landmarks_from_video(
    video_path: str,
    *,
    resize_width: int | None = None,
    min_inference_width: int | None = 1920,
    enhance_luma: bool = True,
    unsharp_amount: float = 0.32,
    start_sec: float = 5.0,
    end_sec: float = 12.0,
    temporal_smooth: float = 0.0,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.22,
    min_tracking_confidence: float = 0.22,
    min_pose_detection_confidence: float | None = None,
    min_pose_presence_confidence: float | None = None,
    min_pose_tracking_confidence: float | None = None,
    min_hand_detection_confidence: float | None = None,
    min_hand_presence_confidence: float | None = None,
    min_hand_tracking_confidence: float | None = None,
    fill_missing_landmarks: bool = True,
    fill_max_gap_frames: int = 2,
    min_xy_coverage_percent: float = 90.0,
    pose_quality: str = "heavy",
    pose_scope: str = "full",
    progress_callback: ProgressCallback | None = None,
) -> ProcessingResult:
    """
    Read a video, run Pose + Hand landmarkers, return a long-form DataFrame.

    Per frame: N pose rows (33 if pose_scope=full, 8 if upper) + 21 left + 21 right.
    Face landmarks are not exported. Missing components use NaN.

    ``timestamp_sec`` is the time of the frame in seconds from the start of the video,
    using the container FPS (or 30 when unknown), matching the MediaPipe video timeline.

    **Time window (export only)** — default matches session cues (START 5 s, STOP 12 s):
      The whole file is still decoded and fed to MediaPipe in order (VIDEO mode needs
      continuity), but CSV rows are written only for frames with
      ``start_sec <= timestamp_sec <= end_sec``. Use ``start_sec=0`` and ``end_sec=float("inf")``
      to export every frame.

    ``model_complexity`` is ignored (kept for API compatibility).

    **Clarity / preprocessing**
      ``pose_quality``: ``"lite"``, ``"full"``, or ``"heavy"`` (default **heavy**, most accurate).
      ``resize_width``: max width after prep; default **None** = no downscale (keeps full
      resolution for sharp landmarks). Set a positive int to cap width for speed.
      ``min_inference_width``: narrow frames are upscaled to at least this width (default **1920**).
      Set ``0`` or ``None`` to disable upscaling.
      ``enhance_luma``: CLAHE on luminance (default **on** for webcam-style clips).
      ``unsharp_amount``: mild unsharp on BGR after CLAHE (default **0.32**); ``0`` disables.
      ``temporal_smooth``: ``0`` = off (raw model output). Use ``0.2``–``0.4`` to reduce jitter
      (slightly laggy motion; good for stable CSVs / visualization).

    ``pose_scope``: ``"full"`` = all **33** body joints; ``"upper"`` = **8** upper-body indices only.

    Optional ``min_*`` overrides default to *None* and fall back to the paired
    ``min_detection_confidence`` / ``min_tracking_confidence``. Hand **presence**
    defaults to the lower of ``0.10`` and ``min_tracking_confidence`` so faint hands
    still track. Defaults are tuned for **maximum landmark recall** on clean video.

    **Coverage guardrail**
      ``fill_missing_landmarks``: interpolate short NaN gaps per landmark id.
      ``fill_max_gap_frames``: only fill runs up to this many frames.
      ``min_xy_coverage_percent``: require at least this percent of rows to have x,y.

    **Adaptive fallback**
      When a frame has weak landmark coverage, extraction runs a second inference pass
      with stronger enhancement and keeps whichever pass has more valid landmarks.
    """
    _ = model_complexity

    try:
        pose_idx_list = pose_indices_for_scope(pose_scope)
    except ValueError as exc:
        return ProcessingResult(
            dataframe=pd.DataFrame(columns=list(COLUMN_ORDER)),
            frames_processed=0,
            ok=False,
            error_message=str(exc),
        )
    expected_rows = landmark_rows_per_frame(pose_scope)

    if not math.isinf(end_sec) and float(end_sec) < float(start_sec):
        return ProcessingResult(
            dataframe=pd.DataFrame(columns=list(COLUMN_ORDER)),
            frames_processed=0,
            ok=False,
            error_message=f"end_sec ({end_sec}) must be >= start_sec ({start_sec}).",
        )

    p_det = (
        min_pose_detection_confidence
        if min_pose_detection_confidence is not None
        else min_detection_confidence
    )
    p_pres = (
        min_pose_presence_confidence
        if min_pose_presence_confidence is not None
        else min(0.15, float(min_detection_confidence))
    )
    p_track = (
        min_pose_tracking_confidence
        if min_pose_tracking_confidence is not None
        else min_tracking_confidence
    )
    h_det = (
        min_hand_detection_confidence
        if min_hand_detection_confidence is not None
        else min_detection_confidence
    )
    h_pres = (
        min_hand_presence_confidence
        if min_hand_presence_confidence is not None
        else min(0.10, float(min_tracking_confidence))
    )
    h_track = (
        min_hand_tracking_confidence
        if min_hand_tracking_confidence is not None
        else min_tracking_confidence
    )

    empty_df = pd.DataFrame(columns=list(COLUMN_ORDER))

    cap: cv2.VideoCapture | None = None
    try:
        cap = cv2.VideoCapture(video_path)
    except Exception as exc:  # noqa: BLE001
        return ProcessingResult(
            dataframe=empty_df.copy(),
            frames_processed=0,
            ok=False,
            error_message=f"Could not open video: {exc}",
        )

    if cap is None or not cap.isOpened():
        return ProcessingResult(
            dataframe=empty_df.copy(),
            frames_processed=0,
            ok=False,
            error_message="Invalid or unreadable video file (failed to open).",
        )

    pose_path, hand_path, model_err = _ensure_pose_and_hand_models(pose_quality)
    if model_err or not pose_path or not hand_path:
        if cap is not None:
            cap.release()
        return ProcessingResult(
            dataframe=empty_df.copy(),
            frames_processed=0,
            ok=False,
            error_message=model_err or "Pose/hand model files are unavailable.",
        )

    total_hint = _frame_total_hint(cap)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1.0:
        fps = 30.0
    ms_per_frame = 1000.0 / fps

    rows: list[
        tuple[int, float, str, int, float, float, float, float]
    ] = []
    frames_read = 0
    frames_exported = 0
    nan = float("nan")

    miw = min_inference_width if (min_inference_width and min_inference_width > 0) else None
    use_smooth = 0.0 < temporal_smooth < 1.0
    pose_state = np.full((len(pose_idx_list), 4), np.nan, dtype=np.float64)
    left_state = np.full((HAND_NUM_LANDMARKS, 4), np.nan, dtype=np.float64)
    right_state = np.full((HAND_NUM_LANDMARKS, 4), np.nan, dtype=np.float64)

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_path),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=p_det,
        min_pose_presence_confidence=p_pres,
        min_tracking_confidence=p_track,
        output_segmentation_masks=False,
    )
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=h_det,
        min_hand_presence_confidence=h_pres,
        min_tracking_confidence=h_track,
    )

    try:
        with (
            PoseLandmarker.create_from_options(pose_options) as pose_lm,
            HandLandmarker.create_from_options(hand_options) as hand_lm,
        ):
            while True:
                ok_read, frame_bgr = cap.read()
                if not ok_read:
                    break
                if frame_bgr is None or frame_bgr.size == 0:
                    break

                frame_bgr = prepare_frame_for_inference(
                    frame_bgr,
                    max_width=resize_width,
                    min_width=miw,
                    enhance_luma=enhance_luma,
                    unsharp_amount=unsharp_amount,
                )
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = np.ascontiguousarray(frame_rgb)
                fh, fw = frame_rgb.shape[0], frame_rgb.shape[1]

                timestamp_ms = int(round(frames_read * ms_per_frame))
                timestamp_sec = timestamp_ms / 1000.0
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=frame_rgb
                )
                pr = pose_lm.detect_for_video(mp_image, timestamp_ms)
                hr = hand_lm.detect_for_video(mp_image, timestamp_ms)

                pose_pts = pr.pose_landmarks[0] if pr.pose_landmarks else []
                n_pt = len(pose_pts)
                raw_pose = np.full((len(pose_idx_list), 4), nan, dtype=np.float64)
                for j, pidx in enumerate(pose_idx_list):
                    if pidx < n_pt and pose_pts[pidx] is not None:
                        raw_pose[j] = _pose_vector(
                            pose_pts[pidx], fw, fh, nan
                        )

                left_lm, right_lm = _assign_left_right_hand_landmarks(hr)
                raw_left = _hand_matrix_from_landmarks(left_lm, fw, fh, nan)
                raw_right = _hand_matrix_from_landmarks(right_lm, fw, fh, nan)

                # Adaptive second pass for harder frames (motion blur / low contrast).
                pose_ok_need = max(1, int(round(0.85 * len(pose_idx_list))))
                first_pose_ok = _count_valid_xy_rows(raw_pose)
                first_hand_ok = _count_valid_xy_rows(raw_left) + _count_valid_xy_rows(raw_right)
                if first_pose_ok < pose_ok_need or first_hand_ok < 30:
                    retry_bgr = prepare_frame_for_inference(
                        frame_bgr,
                        max_width=resize_width,
                        min_width=miw,
                        enhance_luma=True,
                        unsharp_amount=max(float(unsharp_amount), 0.48),
                    )
                    retry_rgb = cv2.cvtColor(retry_bgr, cv2.COLOR_BGR2RGB)
                    retry_rgb = np.ascontiguousarray(retry_rgb)
                    rfh, rfw = retry_rgb.shape[0], retry_rgb.shape[1]
                    retry_img = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=retry_rgb
                    )
                    # Keep monotonically increasing task timestamps.
                    ts_retry = timestamp_ms + 1
                    pr2 = pose_lm.detect_for_video(retry_img, ts_retry)
                    hr2 = hand_lm.detect_for_video(retry_img, ts_retry)

                    pose2_pts = pr2.pose_landmarks[0] if pr2.pose_landmarks else []
                    n2 = len(pose2_pts)
                    raw_pose2 = np.full((len(pose_idx_list), 4), nan, dtype=np.float64)
                    for j, pidx in enumerate(pose_idx_list):
                        if pidx < n2 and pose2_pts[pidx] is not None:
                            raw_pose2[j] = _pose_vector(
                                pose2_pts[pidx], rfw, rfh, nan
                            )
                    left2, right2 = _assign_left_right_hand_landmarks(hr2)
                    raw_left2 = _hand_matrix_from_landmarks(left2, rfw, rfh, nan)
                    raw_right2 = _hand_matrix_from_landmarks(right2, rfw, rfh, nan)

                    score1 = (
                        _count_valid_xy_rows(raw_pose)
                        + _count_valid_xy_rows(raw_left)
                        + _count_valid_xy_rows(raw_right)
                    )
                    score2 = (
                        _count_valid_xy_rows(raw_pose2)
                        + _count_valid_xy_rows(raw_left2)
                        + _count_valid_xy_rows(raw_right2)
                    )
                    if score2 > score1:
                        raw_pose = raw_pose2
                        raw_left = raw_left2
                        raw_right = raw_right2

                if use_smooth:
                    pose_mat = _smooth_landmark_block(
                        raw_pose, pose_state, temporal_smooth
                    )
                else:
                    pose_mat = raw_pose

                in_window = (timestamp_sec + 1e-9 >= float(start_sec)) and (
                    math.isinf(float(end_sec))
                    or timestamp_sec <= float(end_sec) + 1e-9
                )

                if in_window:
                    for j, pidx in enumerate(pose_idx_list):
                        rows.append(
                            _row_pose_from_vector(
                                frames_read,
                                timestamp_sec,
                                pidx,
                                pose_mat[j],
                                nan,
                            )
                        )

                if use_smooth:
                    left_mat = _smooth_landmark_block(
                        raw_left, left_state, temporal_smooth
                    )
                    right_mat = _smooth_landmark_block(
                        raw_right, right_state, temporal_smooth
                    )
                else:
                    left_mat = raw_left
                    right_mat = raw_right

                if in_window:
                    rows.extend(
                        _rows_hand_from_matrix(
                            frames_read,
                            timestamp_sec,
                            left_mat,
                            LANDMARK_TYPE_LEFT,
                            nan,
                        )
                    )
                    rows.extend(
                        _rows_hand_from_matrix(
                            frames_read,
                            timestamp_sec,
                            right_mat,
                            LANDMARK_TYPE_RIGHT,
                            nan,
                        )
                    )
                    frames_exported += 1

                frames_read += 1
                if progress_callback is not None:
                    progress_callback(frames_read, total_hint)
    except cv2.error as exc:
        return ProcessingResult(
            dataframe=empty_df.copy(),
            frames_processed=frames_exported,
            ok=False,
            error_message=f"OpenCV error while reading frames: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return ProcessingResult(
            dataframe=empty_df.copy(),
            frames_processed=frames_exported,
            ok=False,
            error_message=f"Processing failed: {exc}",
        )
    finally:
        if cap is not None:
            cap.release()

    if frames_read == 0:
        return ProcessingResult(
            dataframe=empty_df.copy(),
            frames_processed=0,
            ok=False,
            error_message="Video contains no decodable frames.",
        )

    if frames_exported == 0:
        end_disp = "+inf" if math.isinf(float(end_sec)) else f"{float(end_sec):g}"
        return ProcessingResult(
            dataframe=empty_df.copy(),
            frames_processed=0,
            ok=False,
            error_message=(
                f"No frames in export window [{float(start_sec):g}, {end_disp}] s "
                f"({frames_read} frame(s) read; video may be shorter than start_sec)."
            ),
        )

    df = pd.DataFrame(rows, columns=list(COLUMN_ORDER))
    df["frame_id"] = df["frame_id"].astype(np.int64)
    df["timestamp_sec"] = df["timestamp_sec"].astype(np.float64)
    df["landmark_type"] = df["landmark_type"].astype("string")
    df["landmark_id"] = df["landmark_id"].astype(np.int64)
    for col in ("x", "y", "z", "visibility"):
        df[col] = df[col].astype(np.float64)

    before_cov = _xy_coverage_percent(df)
    if fill_missing_landmarks:
        df = _fill_missing_landmark_gaps(df, max_gap=fill_max_gap_frames)
        for col in ("x", "y", "z", "visibility"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
    after_cov = _xy_coverage_percent(df)

    if len(df) != frames_exported * expected_rows:
        return ProcessingResult(
            dataframe=empty_df.copy(),
            frames_processed=frames_exported,
            ok=False,
            error_message=(
                "Internal validation failed: row count does not match "
                f"{expected_rows} rows per frame (pose_scope={pose_scope!r})."
            ),
        )

    need_cov = max(0.0, min(100.0, float(min_xy_coverage_percent)))
    if after_cov + 1e-9 < need_cov:
        return ProcessingResult(
            dataframe=empty_df.copy(),
            frames_processed=frames_exported,
            ok=False,
            error_message=(
                f"Landmark coverage too low: {after_cov:.1f}% x/y present "
                f"(before fill {before_cov:.1f}%, required >= {need_cov:.1f}%). "
                "Improve framing/lighting or relax thresholds."
            ),
        )

    return ProcessingResult(
        dataframe=df,
        frames_processed=frames_exported,
        ok=True,
        error_message=None,
    )


def write_landmarks_csv(df: pd.DataFrame, path: str | Path) -> None:
    """
    Write the landmark table to disk without losing float precision.

    Use this instead of bare ``to_csv`` if Excel or other tools show rounded numbers.
    ``NaN`` means that landmark was missing for that frame (occlusion / low confidence).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        p,
        index=False,
        encoding="utf-8",
        # Fixed decimals for x,y,z,visibility — easier to read and compare than short rounding.
        float_format="%.10f",
        na_rep="NaN",
    )


VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _list_videos(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
            out.append(p)
    return out


def cli_main() -> int:
    """Single-file CLI: extract pose+hands landmarks to CSV."""
    import argparse
    import sys

    root = _project_root()
    default_in = root / "videos"
    default_out = root / "landmark_output"

    parser = argparse.ArgumentParser(
        description="Extract full-body pose + both hands to CSV (33 + 21 + 21 rows/frame by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", type=Path, default=None, help="Process one file only.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (only with --video; default: <output-dir>/<name>_landmarks.csv).",
    )
    parser.add_argument("--input-dir", type=Path, default=default_in, help="Input folder (ignored if --video set).")
    parser.add_argument("--output-dir", type=Path, default=default_out, help="Folder for CSV outputs.")
    parser.add_argument(
        "--resize-width",
        type=int,
        default=0,
        help="Max width after prep; 0 = no downscale (best landmark detail).",
    )
    parser.add_argument(
        "--min-inference-width",
        type=int,
        default=1920,
        help="Upscale frames narrower than this width (0 = off).",
    )
    parser.add_argument(
        "--no-enhance-luma",
        action="store_true",
        help="Disable CLAHE (default is on for clearer landmarks in typical lighting).",
    )
    parser.add_argument(
        "--unsharp",
        type=float,
        default=0.32,
        metavar="K",
        help="Unsharp strength after CLAHE (0 disables).",
    )
    parser.add_argument("--temporal-smooth", type=float, default=0.0, metavar="LAMBDA", help="0..1 smoothing; 0=raw.")
    parser.add_argument("--pose-quality", choices=("lite", "full", "heavy"), default="heavy")
    parser.add_argument("--pose-scope", choices=("full", "upper"), default="full")
    parser.add_argument(
        "--min-detection",
        type=float,
        default=0.22,
        help="Pose + hand detection floor (lower = more landmarks, noisier).",
    )
    parser.add_argument(
        "--min-tracking",
        type=float,
        default=0.22,
        help="Tracking floor for pose + hands.",
    )
    parser.add_argument("--min-pose-presence", type=float, default=None)
    parser.add_argument(
        "--min-hand-presence",
        type=float,
        default=0.08,
        help="Hand presence threshold (lower keeps weak hand tracks).",
    )
    parser.add_argument(
        "--no-fill-missing",
        action="store_true",
        help="Disable interpolation/fill for missing landmark gaps.",
    )
    parser.add_argument(
        "--fill-max-gap",
        type=int,
        default=2,
        metavar="N",
        help="Fill only gaps up to N frames (used only when filling enabled).",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=90.0,
        metavar="PCT",
        help="Required minimum x/y coverage percentage in output CSV.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=5.0,
        metavar="T",
        help="Only write CSV rows for frames at or after T seconds (full video still decoded).",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=12.0,
        metavar="T",
        help="Only write CSV rows for frames at or before T seconds (inclusive).",
    )
    parser.add_argument(
        "--export-full-video",
        action="store_true",
        help="Export every frame (overrides --start-sec / --end-sec).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.video is not None:
        vp = args.video.expanduser().resolve()
        if not vp.is_file():
            print(f"Not a file: {vp}", file=sys.stderr)
            return 1
        if vp.suffix.lower() not in VIDEO_SUFFIXES:
            print(f"Unsupported extension {vp.suffix!r}; use one of: {VIDEO_SUFFIXES}", file=sys.stderr)
            return 1
        videos = [vp]
        single_out: Path | None = None
        if args.out is not None:
            single_out = args.out.expanduser().resolve()
            single_out.parent.mkdir(parents=True, exist_ok=True)
    else:
        single_out = None
        if args.out is not None:
            print("--out is only valid with --video.", file=sys.stderr)
            return 1
        input_dir = args.input_dir.expanduser().resolve()
        input_dir.mkdir(parents=True, exist_ok=True)
        videos = _list_videos(input_dir)
        if not videos:
            print(f"No video files found in: {input_dir}", file=sys.stderr)
            return 1

    rw = None if args.resize_width == 0 else args.resize_width
    miw = None if args.min_inference_width == 0 else args.min_inference_width
    ts = float(args.temporal_smooth)
    if ts < 0.0 or ts > 1.0:
        print("--temporal-smooth must be between 0 and 1.", file=sys.stderr)
        return 1
    enhance_luma = not bool(args.no_enhance_luma)
    unsharp = max(0.0, float(args.unsharp))

    if args.export_full_video:
        win_start, win_end = 0.0, float("inf")
    else:
        win_start, win_end = float(args.start_sec), float(args.end_sec)
        if not math.isinf(win_end) and win_end < win_start:
            print("ERROR: --end-sec must be >= --start-sec.", file=sys.stderr)
            return 1

    failed = 0
    for i, vid in enumerate(videos, 1):
        if single_out is not None and len(videos) == 1:
            out_csv = single_out
        else:
            out_csv = output_dir / f"{vid.stem}_landmarks.csv"
        print(f"[{i}/{len(videos)}] {vid.name} -> {out_csv.name}")
        end_disp = "+inf" if math.isinf(win_end) else f"{win_end:g}"
        print(f"  export window: [{win_start:g}, {end_disp}] s of source timeline")

        def on_progress(done: int, total: int | None) -> None:
            if total and total > 0:
                pct = 100.0 * min(done / total, 1.0)
                print(f"  frames {done}/{total} ({pct:.1f}%)", end="\r", flush=True)
            else:
                print(f"  frames {done}", end="\r", flush=True)

        extract_kw: dict = {
            "resize_width": rw,
            "min_inference_width": miw,
            "enhance_luma": enhance_luma,
            "unsharp_amount": unsharp,
            "temporal_smooth": ts,
            "min_detection_confidence": args.min_detection,
            "min_tracking_confidence": args.min_tracking,
            "pose_quality": args.pose_quality,
            "pose_scope": args.pose_scope,
            "progress_callback": on_progress,
            "start_sec": win_start,
            "end_sec": win_end,
            "fill_missing_landmarks": not bool(args.no_fill_missing),
            "fill_max_gap_frames": max(0, int(args.fill_max_gap)),
            "min_xy_coverage_percent": float(args.min_coverage),
        }
        if args.min_pose_presence is not None:
            extract_kw["min_pose_presence_confidence"] = args.min_pose_presence
        extract_kw["min_hand_presence_confidence"] = float(args.min_hand_presence)

        res = extract_pose_landmarks_from_video(str(vid), **extract_kw)
        print()
        if not res.ok:
            print(f"  ERROR: {res.error_message}", file=sys.stderr)
            failed += 1
            continue
        write_landmarks_csv(res.dataframe, out_csv)
        cov = _xy_coverage_percent(res.dataframe)
        print(f"  OK: {res.frames_processed} frames, {len(res.dataframe)} rows | coverage={cov:.1f}%")

    if failed:
        print(f"Finished with {failed} failure(s).", file=sys.stderr)
        return 2
    print("All videos processed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())

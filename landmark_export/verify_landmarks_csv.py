#!/usr/bin/env python3
"""
Verify exported landmark CSVs: structure + optional visual overlay on the source video.

Structural checks
-----------------
- Expected columns (frame_id, timestamp_sec, landmark_type, landmark_id, x, y, z, visibility)
- Rows per ``frame_id`` match the export: **50** (8 pose + 21 + 21) for upper-body
- Pose / hand ``landmark_id`` sets match that mode; scope is inferred from the first frame that contains pose rows
- x, y roughly in normalized image space (wide bounds; limbs often extend slightly past the frame)


Usage
-----
  python landmark_export/verify_landmarks_csv.py path/to/file_landmarks.csv


"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_LE = Path(__file__).resolve().parent
if str(_LE) not in sys.path:
    sys.path.insert(0, str(_LE))

from holistic_pose_utils import (  # noqa: E402
    COLUMN_ORDER,
    LANDMARK_TYPE_LEFT,
    LANDMARK_TYPE_POSE,
    LANDMARK_TYPE_RIGHT,
    FULL_BODY_POSE_NUM,
    POSE_LANDMARK_INDICES,
    landmark_rows_per_frame,
    prepare_frame_for_inference,
)

# Only these CSV landmark_type values are drawn (whatever x,y the CSV contains).
CSV_DRAW_TYPES = frozenset(
    (LANDMARK_TYPE_POSE, LANDMARK_TYPE_LEFT, LANDMARK_TYPE_RIGHT)
)

# Draw only the same compact upper-body pose subset used by runtime detector.
VERIFY_POSE_DRAW_IDS: frozenset[int] = frozenset(POSE_LANDMARK_INDICES)

POSE_CONNECTIONS_UPPER: tuple[tuple[int, int], ...] = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
)

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def _prep_like_extraction(
    frame_bgr: np.ndarray,
    *,
    resize_width: int | None,
    min_inference_width: int | None,
    enhance_luma: bool,
    unsharp_amount: float = 0.0,
) -> np.ndarray:
    """Must mirror ``extract_pose_landmarks_from_video`` frame preprocessing."""
    return prepare_frame_for_inference(
        frame_bgr,
        max_width=resize_width,
        min_width=min_inference_width,
        enhance_luma=enhance_luma,
        unsharp_amount=unsharp_amount,
    )


def _infer_pose_scope_from_df(df: pd.DataFrame) -> tuple[str, set[int], int] | None:
    """Return (scope, expected_pose_ids, expected_rows_per_frame) or None if no pose rows."""
    for fid in sorted(df["frame_id"].unique()):
        pose0 = df[
            (df["frame_id"] == fid) & (df["landmark_type"] == LANDMARK_TYPE_POSE)
        ]
        if pose0.empty:
            continue
        ids = set(pose0["landmark_id"].astype(int))
        if ids == set(range(FULL_BODY_POSE_NUM)):
            return "full", ids, landmark_rows_per_frame("full")
        if ids == set(POSE_LANDMARK_INDICES):
            return "upper", ids, landmark_rows_per_frame("upper")
        return (
            "unknown",
            ids,
            len(ids) + 21 * 2,
        )
    return None


def _structural_report(df: pd.DataFrame) -> tuple[bool, list[str]]:
    issues: list[str] = []
    ok = True

    missing = [c for c in COLUMN_ORDER if c not in df.columns]
    if missing:
        ok = False
        issues.append(f"Missing columns: {missing}")

    extra = [c for c in df.columns if c not in COLUMN_ORDER]
    if extra:
        issues.append(f"Extra columns (ignored in strict check): {extra}")

    if "frame_id" not in df.columns:
        return False, issues

    inferred = _infer_pose_scope_from_df(df)
    if inferred is None:
        ok = False
        issues.append("No pose rows found; cannot infer export layout.")
        n_frames = int(df["frame_id"].max()) + 1 if len(df) else 0
        issues.insert(0, f"Rows: {len(df)} | distinct frames: {df['frame_id'].nunique()} | max frame_id+1: {n_frames}")
        return ok, issues

    scope, expected_pose, expected_rows = inferred
    if scope == "unknown":
        ok = False
        issues.append(
            "Unrecognized pose landmark_id set "
            f"(expected 0..32 full-body or {sorted(POSE_LANDMARK_INDICES)} upper). "
            f"Example: {sorted(expected_pose)[:20]}..."
        )
    else:
        g = df.groupby("frame_id", sort=True).size()
        bad = g[g != expected_rows]
        if len(bad) > 0:
            ok = False
            sample = bad.head(5).to_dict()
            issues.append(
                f"{len(bad)} frame_id(s) do not have {expected_rows} rows ({scope} export) "
                f"(sample): {sample}"
            )

        for ftype, expected_ids in (
            (LANDMARK_TYPE_POSE, expected_pose),
            (LANDMARK_TYPE_LEFT, set(range(21))),
            (LANDMARK_TYPE_RIGHT, set(range(21))),
        ):
            sub = df[df["landmark_type"] == ftype]
            if sub.empty and ftype == LANDMARK_TYPE_POSE:
                continue
            by_f = sub.groupby("frame_id")["landmark_id"].apply(set)
            wrong = by_f[by_f != expected_ids]
            if len(wrong) > 0:
                ok = False
                issues.append(
                    f"{ftype}: wrong landmark_id set in {len(wrong)} frames (e.g. frame "
                    f"{wrong.index[0]} got {wrong.iloc[0]})"
                )
                break

    # MediaPipe-normalized x,y: count only non-NaN cells (missing hands/pose gaps are NaN, not out of range).
    for col in ("x", "y"):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        v = s.dropna()
        if len(v) == 0:
            continue
        inside = (v >= -0.5) & (v <= 1.5)
        pct = 100.0 * float(inside.sum()) / float(len(v))
        if pct < 90.0:
            issues.append(
                f"Column {col}: only {pct:.1f}% of non-NaN values in ~[-0.5, 1.5] (normalized); "
                "if overlays look wrong, match --resize-width to extraction."
            )

    n_frames = int(df["frame_id"].max()) + 1 if len(df) else 0
    summary = (
        f"Rows: {len(df)} | distinct frames: {df['frame_id'].nunique()} | "
        f"max frame_id+1: {n_frames} | inferred pose scope: {scope}"
    )
    issues.insert(0, summary)

    return ok, issues


def _draw_frame_overlay(
    frame_bgr: np.ndarray,
    frame_df: pd.DataFrame,
) -> np.ndarray:
    """Draw compact upper-body pose + both hands skeleton on white background."""
    h, w = frame_bgr.shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    if frame_df.empty or "landmark_type" not in frame_df.columns:
        return out

    plot = frame_df[frame_df["landmark_type"].astype(str).isin(CSV_DRAW_TYPES)].copy()
    if plot.empty:
        return out

    lid = pd.to_numeric(plot["landmark_id"], errors="coerce")
    is_pose = plot["landmark_type"].astype(str) == LANDMARK_TYPE_POSE
    plot = plot[~is_pose | lid.isin(VERIFY_POSE_DRAW_IDS)]
    if plot.empty:
        return out

    def pt_xy(row) -> tuple[int, int] | None:
        if pd.isna(row["x"]) or pd.isna(row["y"]):
            return None
        x = float(row["x"])
        y = float(row["y"])
        xi = int(round(x * (w - 1)))
        yi = int(round(y * (h - 1)))
        return xi, yi

    colors = {
        LANDMARK_TYPE_POSE: (0, 255, 0),
        LANDMARK_TYPE_LEFT: (255, 128, 0),
        LANDMARK_TYPE_RIGHT: (0, 128, 255),
    }
    point_map: dict[tuple[str, int], tuple[int, int]] = {}
    for _, row in plot.iterrows():
        p = pt_xy(row)
        if p is None:
            continue
        lt = str(row["landmark_type"])
        lid_int = int(row["landmark_id"])
        point_map[(lt, lid_int)] = p
        color = colors[lt]
        cv2.circle(out, p, 3, color, -1)

    # Pose upper-body connections.
    for a, b in POSE_CONNECTIONS_UPPER:
        pa = point_map.get((LANDMARK_TYPE_POSE, a))
        pb = point_map.get((LANDMARK_TYPE_POSE, b))
        if pa is not None and pb is not None:
            cv2.line(out, pa, pb, colors[LANDMARK_TYPE_POSE], 2)

    # Hand skeleton connections.
    for hand_type in (LANDMARK_TYPE_LEFT, LANDMARK_TYPE_RIGHT):
        color = colors[hand_type]
        for a, b in HAND_CONNECTIONS:
            pa = point_map.get((hand_type, a))
            pb = point_map.get((hand_type, b))
            if pa is not None and pb is not None:
                cv2.line(out, pa, pb, color, 2)

    return out


def _prepare_landmark_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure frame_id is integer for matching video frame indices."""
    out = df.copy()
    out["frame_id"] = pd.to_numeric(out["frame_id"], errors="coerce").astype("Int64")
    return out


def _write_sample_overlays(
    df: pd.DataFrame,
    video_path: Path,
    out_dir: Path,
    num_samples: int,
    resize_width: int | None,
    min_inference_width: int | None,
    enhance_luma: bool,
    unsharp_amount: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_fids = df["frame_id"].dropna()
    if valid_fids.empty:
        print("No frames in CSV.", file=sys.stderr)
        return

    max_f = int(valid_fids.max())
    if max_f < 0:
        print("No frames in CSV.", file=sys.stderr)
        return

    indices = sorted(
        {
            int(round(t))
            for t in np.linspace(0, max_f, num=min(num_samples, max_f + 1), dtype=float)
        }
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}", file=sys.stderr)
        return

    for fid in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"Could not read frame {fid}", file=sys.stderr)
            continue

        frame_proc = _prep_like_extraction(
            frame,
            resize_width=resize_width,
            min_inference_width=min_inference_width,
            enhance_luma=enhance_luma,
            unsharp_amount=unsharp_amount,
        )
        sub = df[df["frame_id"] == fid]
        vis = _draw_frame_overlay(frame_proc, sub)
        outp = out_dir / f"overlay_frame_{fid:06d}.png"
        cv2.imwrite(str(outp), vis)
        print(f"Wrote {outp}")

    cap.release()


def _write_overlay_video(
    df: pd.DataFrame,
    video_path: Path,
    out_video: Path,
    resize_width: int | None,
    min_inference_width: int | None,
    enhance_luma: bool,
    unsharp_amount: float,
    progress_every: int = 30,
) -> bool:
    """Encode full video with CSV landmarks (pose + hands only) drawn on each frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}", file=sys.stderr)
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    ret, frame0 = cap.read()
    if not ret or frame0 is None:
        print("Could not read first video frame.", file=sys.stderr)
        cap.release()
        return False

    proc0 = _prep_like_extraction(
        frame0,
        resize_width=resize_width,
        min_inference_width=min_inference_width,
        enhance_luma=enhance_luma,
        unsharp_amount=unsharp_amount,
    )
    h, w = proc0.shape[:2]
    out_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Could not open VideoWriter for {out_video}", file=sys.stderr)
        cap.release()
        return False

    sub0 = df[df["frame_id"] == 0]
    writer.write(_draw_frame_overlay(proc0, sub0))
    n_out = 1
    fid = 1
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        proc = _prep_like_extraction(
            frame,
            resize_width=resize_width,
            min_inference_width=min_inference_width,
            enhance_luma=enhance_luma,
            unsharp_amount=unsharp_amount,
        )
        sub = df[df["frame_id"] == fid]
        writer.write(_draw_frame_overlay(proc, sub))
        n_out += 1
        fid += 1
        if progress_every and fid % progress_every == 0:
            print(f"  encoded {fid} frames...")

    cap.release()
    writer.release()
    print(f"Wrote overlaid video: {out_video} ({n_out} frames, {fps:.2f} fps)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify landmark CSV structure and optional overlays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  %(prog)s landmark_output/clip_landmarks.csv\n"
        "  %(prog)s out.csv --video clip.mp4 --out-dir ./overlays\n",
    )
    parser.add_argument(
        "csv_pos",
        nargs="?",
        type=Path,
        default=None,
        metavar="CSV",
        help="Path to landmarks CSV (positional).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        dest="csv_opt",
        metavar="PATH",
        help="Path to landmarks CSV (alternative to positional CSV).",
    )
    parser.add_argument("--video", type=Path, default=None, help="Source video (same as used to build CSV)")
    parser.add_argument(
        "--out-video",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write full overlaid MP4 (pose + hands from CSV only). Requires --video.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Folder for sample PNG overlays (requires --video). Default if --num-samples > 0.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Sample PNG count; use 0 to skip PNGs when only --out-video is needed.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=0,
        help="Same as extraction (0 = no downscale); match extraction for overlays.",
    )
    parser.add_argument(
        "--min-inference-width",
        type=int,
        default=1920,
        help="Same as extraction (default 1920); 0 = no upscaling step.",
    )
    parser.add_argument(
        "--no-enhance-luma",
        action="store_true",
        help="Disable CLAHE (extraction uses CLAHE by default).",
    )
    parser.add_argument(
        "--unsharp",
        type=float,
        default=0.32,
        metavar="K",
        help="Same unsharp K as extraction (0 disables).",
    )
    args = parser.parse_args()

    csv_path = args.csv_opt or args.csv_pos
    if csv_path is None:
        parser.error(
            "missing CSV path. Examples:\n"
            f"  {parser.prog} landmark_output/clip_landmarks.csv\n"
            f"  {parser.prog} --csv landmark_output/clip_landmarks.csv"
        )
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    df = _prepare_landmark_df(df)
    ok, issues = _structural_report(df)
    print("=== Structural check ===")
    for line in issues:
        print(line)
    print("=== Result ===")
    if ok:
        print("PASS: layout and per-frame row counts look consistent.")
    else:
        print("FAIL: see issues above.")
        return 2

    if args.video is not None:
        vp = args.video.expanduser().resolve()
        if not vp.is_file():
            print(f"Video not found: {vp}", file=sys.stderr)
            return 1
        rw = args.resize_width
        if rw == 0:
            rw = None
        miw = args.min_inference_width
        if miw == 0:
            miw = None
        enhance_luma = not bool(args.no_enhance_luma)
        unsharp = max(0.0, float(args.unsharp))

        if args.out_video is not None:
            outp = args.out_video.expanduser().resolve()
            print("\n=== Overlay video (CSV pose + hands only) ===")
            print(
                f"resize_width={rw!r}, min_inference_width={miw!r}, "
                f"enhance_luma={enhance_luma}, unsharp={unsharp} (must match extraction)."
            )
            if not _write_overlay_video(
                df, vp, outp, rw, miw, enhance_luma, unsharp
            ):
                return 1

        if args.num_samples > 0:
            out_dir = args.out_dir or (csv_path.parent / "verify_overlays")
            print("\n=== Sample PNG overlays ===")
            print(
                f"Saving ~{args.num_samples} frames to {out_dir} | "
                f"resize_width={rw!r}, min_inference_width={miw!r}, "
                f"enhance_luma={enhance_luma}, unsharp={unsharp}"
            )
            _write_sample_overlays(
                df, vp, out_dir, args.num_samples, rw, miw, enhance_luma, unsharp
            )
            print("Green = pose, orange = left hand, blue = right hand.")
        elif args.out_video is None:
            print("\nTip: use --out-video out.mp4 for a full verified video, or --num-samples 8 for PNGs.")
    else:
        if args.out_video is not None:
            print("--out-video requires --video.", file=sys.stderr)
            return 1
        print("\nTip: add --video and --out-video out.mp4 to draw CSV landmarks on the full clip.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

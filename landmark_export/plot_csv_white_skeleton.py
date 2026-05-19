#!/usr/bin/env python3
"""
Render CSV landmarks to a white-background skeleton video (colored lines/points).

This does NOT redraw on the source video. It only uses:
- pose upper-body points: 11,12,13,14,15,16,23,24
- left_hand (21)
- right_hand (21)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
POSE_IDS = (11, 12, 13, 14, 15, 16, 23, 24)
POSE_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def _to_px(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    return int(round(x * (w - 1))), int(round(y * (h - 1)))


def _draw_frame(canvas: np.ndarray, frame_df: pd.DataFrame) -> np.ndarray:
    out = canvas.copy()
    h, w = out.shape[:2]
    pts: dict[tuple[str, int], tuple[int, int]] = {}

    for _, row in frame_df.iterrows():
        lt = str(row["landmark_type"])
        lid = int(row["landmark_id"])
        if lt == "pose" and lid not in POSE_IDS:
            continue
        if lt not in ("pose", "left_hand", "right_hand"):
            continue
        x = row["x"]
        y = row["y"]
        if pd.isna(x) or pd.isna(y):
            continue
        p = _to_px(float(x), float(y), w, h)
        pts[(lt, lid)] = p

    point_colors = {
        "pose": (0, 255, 0),        # green
        "left_hand": (255, 0, 0),   # blue (BGR)
        "right_hand": (0, 165, 255) # orange (BGR)
    }
    line_color = (0, 0, 0)  # black connection lines for all skeleton parts
    for a, b in POSE_CONNECTIONS:
        pa = pts.get(("pose", a))
        pb = pts.get(("pose", b))
        if pa is not None and pb is not None:
            cv2.line(out, pa, pb, line_color, 2)
    for hand in ("left_hand", "right_hand"):
        for a, b in HAND_CONNECTIONS:
            pa = pts.get((hand, a))
            pb = pts.get((hand, b))
            if pa is not None and pb is not None:
                cv2.line(out, pa, pb, line_color, 2)
    for key, p in pts.items():
        lt, _ = key
        cv2.circle(out, p, 3, point_colors[lt], -1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot CSV landmarks to white-background skeleton video.",
    )
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV path.")
    parser.add_argument("--out-video", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument("--video", type=Path, default=None, help="Optional source video (for size/fps/length).")
    parser.add_argument("--width", type=int, default=1280, help="Canvas width if --video is not given.")
    parser.add_argument("--height", type=int, default=720, help="Canvas height if --video is not given.")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS if --video is not given.")
    parser.add_argument(
        "--preserve-source-timeline",
        action="store_true",
        help=(
            "Write frame 0..max(frame_id) with blanks where CSV has no points. "
            "Default writes only CSV frame_ids (best for 5s–12s task windows)."
        ),
    )
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("CSV is empty.", file=sys.stderr)
        return 1

    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce").astype("Int64")
    valid_fids = df["frame_id"].dropna().astype(int)
    if valid_fids.empty:
        print("CSV has no valid frame_id values.", file=sys.stderr)
        return 1
    frame_ids_sorted = sorted(set(int(x) for x in valid_fids.tolist()))
    min_fid = frame_ids_sorted[0]
    max_fid = frame_ids_sorted[-1]

    width = args.width
    height = args.height
    fps = float(args.fps)
    n_frames = max_fid + 1

    if args.video is not None:
        vp = args.video.expanduser().resolve()
        cap = cv2.VideoCapture(str(vp))
        if cap.isOpened():
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            vf = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            vt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if vw > 0 and vh > 0:
                width, height = vw, vh
            if vf > 1.0:
                fps = vf
            if vt > 0:
                n_frames = min(n_frames, vt)

    args.out_video.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out_video.expanduser().resolve()),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        print(f"Could not open output video: {args.out_video}", file=sys.stderr)
        return 1

    base = np.full((height, width, 3), 255, dtype=np.uint8)
    if args.preserve_source_timeline:
        for fid in range(n_frames):
            sub = df[df["frame_id"] == fid]
            writer.write(_draw_frame(base, sub))
        mode_msg = "source timeline"
        out_frames = n_frames
    else:
        # Default: render only the extracted window (e.g., 5s–12s), no blank lead/trail.
        for fid in frame_ids_sorted:
            sub = df[df["frame_id"] == fid]
            writer.write(_draw_frame(base, sub))
        mode_msg = "csv-window only"
        out_frames = len(frame_ids_sorted)
    writer.release()
    print(
        f"Wrote white skeleton video: {args.out_video} "
        f"({out_frames} frames @ {fps:.2f} fps, {mode_msg}, "
        f"frame_id range {min_fid}..{max_fid})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

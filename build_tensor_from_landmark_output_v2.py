from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Column order in CSV / internal 4-vector (x, y, z, visibility).
COORDS = ("x", "y", "z", "visibility")

# Align with holistic_pose_utils: full = 33 pose + hands; upper = 8 upper-body pose ids + hands (50 rows/frame).
POSE_LANDMARK_INDICES_UPPER: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 23, 24)


def _lm_order_for_pose_scope(scope: str) -> list[tuple[str, int]]:
    s = scope.lower().strip()
    if s == "full":
        return (
            [("pose", i) for i in range(33)]
            + [("left_hand", i) for i in range(21)]
            + [("right_hand", i) for i in range(21)]
        )
    if s in ("upper", "upper_body"):
        return (
            [("pose", i) for i in POSE_LANDMARK_INDICES_UPPER]
            + [("left_hand", i) for i in range(21)]
            + [("right_hand", i) for i in range(21)]
        )
    raise ValueError(f"pose_scope must be 'full' or 'upper', got {scope!r}")


def _normalize_task_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"_landmarks$", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("bursh", "brush")
    s = s.replace("stacking_book", "stacking books")
    if s == "carrying bag":
        s = "carrying the bag"
    if s == "tying":
        s = "tying shoe"
    return s


def _coord_indices_from_preset(preset: str) -> tuple[int, ...]:
    p = preset.lower().strip()
    if p in ("four", "4", "xyzv", "all"):
        return (0, 1, 2, 3)
    if p in ("xyv", "3", "xyvis", "xy_visibility"):
        return (0, 1, 3)
    raise ValueError(f"coord_preset must be 'four' or 'xyv', got {preset!r}")


def _parse_task_and_rep(csv_path: Path) -> tuple[str, int] | None:
    stem = csv_path.stem
    if not stem.endswith("_landmarks"):
        return None
    name = stem[: -len("_landmarks")]
    m = re.match(r"^(.*?)(\d+)$", name.strip())
    if m:
        return _normalize_task_name(m.group(1).strip()), int(m.group(2))
    return _normalize_task_name(name.strip()), 1


def _count_frames(csv_path: Path) -> int:
    df = pd.read_csv(csv_path, usecols=["frame_id"])
    return int(df["frame_id"].nunique())


def _load_clip(
    csv_path: Path,
    target_frames: int,
    lm_index: dict[tuple[str, int], int],
    num_landmarks: int,
    coord_idx: tuple[int, ...],
) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce").astype("Int64")
    df["landmark_id"] = pd.to_numeric(df["landmark_id"], errors="coerce").astype("Int64")
    for c in COORDS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    frame_ids = sorted(df["frame_id"].dropna().astype(int).unique().tolist())
    if not frame_ids:
        raise ValueError(f"No valid frame_id values: {csv_path}")

    clip = np.full((len(frame_ids), num_landmarks, 4), np.nan, dtype=np.float64)
    for fi, fid in enumerate(frame_ids):
        sub = df[df["frame_id"] == fid]
        for _, r in sub.iterrows():
            key = (str(r["landmark_type"]), int(r["landmark_id"]))
            li = lm_index.get(key)
            if li is None:
                continue
            clip[fi, li, 0] = r["x"]
            clip[fi, li, 1] = r["y"]
            clip[fi, li, 2] = r["z"]
            clip[fi, li, 3] = r["visibility"]

    c_out = len(coord_idx)
    src = np.linspace(0, len(frame_ids) - 1, num=target_frames)
    out = np.empty((target_frames, num_landmarks, c_out), dtype=np.float64)
    full = np.empty((target_frames, num_landmarks, 4), dtype=np.float64)
    for t, s in enumerate(src):
        lo = int(np.floor(s))
        hi = min(lo + 1, len(frame_ids) - 1)
        a = s - lo
        full[t] = (1.0 - a) * clip[lo] + a * clip[hi]
        out[t] = full[t][..., list(coord_idx)]
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build (Tasks,Reps,Frames,Landmarks,Coords) 5D tensor from landmark CSVs.")
    p.add_argument("--csv-dir", type=Path, default=Path("/Users/battulamonika/PycharmProjects/armtracking/landmark_output"))
    p.add_argument(
        "--out-npy",
        type=Path,
        default=Path("/Users/battulamonika/PycharmProjects/armtracking/rpca_input_xyv_upper106.npy"),
        help="5D tensor path for 5_RPCA.py (T,R,F,L,C). Default matches the project RPCA input.",
    )
    p.add_argument(
        "--pose-scope",
        type=str,
        default="upper",
        choices=("full", "upper"),
        help="Must match CSV export: upper = 8 pose + 21 + 21 = 50 landmarks; full = 33 + 21 + 21 = 75.",
    )
    p.add_argument(
        "--target-frames",
        type=int,
        default=0,
        help="Resampled clip length F. Use 0 to set F = max unique frame_id count over all selected CSVs.",
    )
    p.add_argument(
        "--coord-preset",
        type=str,
        default="xyv",
        choices=("xyv", "four"),
        help="xyv = (x,y,visibility) three channels; four = (x,y,z,visibility).",
    )
    p.add_argument("--require-reps", type=int, default=3)
    p.add_argument("--missing-rep-policy", type=str, default="duplicate", choices=("duplicate", "skip"))
    p.add_argument("--task-list", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lm_order = _lm_order_for_pose_scope(args.pose_scope)
    lm_index = {k: i for i, k in enumerate(lm_order)}
    num_landmarks = len(lm_order)

    csv_dir = args.csv_dir.expanduser().resolve()
    csv_files = sorted(csv_dir.glob("*_landmarks.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_landmarks.csv files in {csv_dir}")

    by_task: dict[str, dict[int, Path]] = {}
    for p in csv_files:
        parsed = _parse_task_and_rep(p)
        if parsed is None:
            continue
        task, rep = parsed
        by_task.setdefault(task, {})
        prev = by_task[task].get(rep)
        if prev is None or p.stat().st_size > prev.stat().st_size:
            by_task[task][rep] = p

    if args.task_list is not None:
        raw = args.task_list.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        task_order = [_normalize_task_name(x) for x in raw if x.strip()]
    else:
        task_order = sorted(by_task.keys())

    print("Detected tasks/reps:")
    for t in sorted(by_task):
        print(f"  - {t}: reps={sorted(by_task[t].keys())}")

    selected: list[tuple[str, list[Path]]] = []
    fill_notes: list[str] = []
    reps_needed = list(range(1, int(args.require_reps) + 1))
    for t in task_order:
        rep_map = by_task.get(t, {})
        present = sorted(rep_map.keys())
        if not present:
            continue
        if all(r in rep_map for r in reps_needed):
            selected.append((t, [rep_map[r] for r in reps_needed]))
            continue
        if args.missing_rep_policy == "skip":
            print(f"Skipping task (missing reps 1..{args.require_reps}): {t}")
            continue
        chosen: list[Path] = []
        for r in reps_needed:
            if r in rep_map:
                chosen.append(rep_map[r])
            else:
                lower = [x for x in present if x < r]
                fallback = max(lower) if lower else min(present)
                chosen.append(rep_map[fallback])
                fill_notes.append(f"{t}: rep{r} <- rep{fallback} ({rep_map[fallback].name})")
        selected.append((t, chosen))

    if args.dry_run:
        print("\nDry run only. Selected tasks:")
        for t, paths in selected:
            print(f"  {t}")
            for p in paths:
                print(f"    {p.name}")
        if fill_notes:
            print("\nFilled missing reps:")
            for n in fill_notes:
                print(f"  - {n}")
        return

    if not selected:
        raise RuntimeError("No tasks selected for tensor creation.")

    coord_idx = _coord_indices_from_preset(args.coord_preset)
    C = len(coord_idx)

    if int(args.target_frames) <= 0:
        F = 0
        for _, rep_paths in selected:
            for p in rep_paths:
                F = max(F, _count_frames(p))
        if F <= 0:
            raise RuntimeError("Auto target-frames failed (F<=0).")
        print(f"Auto target_frames={F} (max unique frame_id over selected CSVs).")
    else:
        F = int(args.target_frames)

    T, R = len(selected), int(args.require_reps)
    tensor = np.zeros((T, R, F, num_landmarks, C), dtype=np.float32)

    for ti, (task, rep_paths) in enumerate(selected):
        for ri, p in enumerate(rep_paths):
            tensor[ti, ri] = _load_clip(
                p,
                target_frames=F,
                lm_index=lm_index,
                num_landmarks=num_landmarks,
                coord_idx=coord_idx,
            )
        print(f"Loaded task {ti:02d}: {task}")
    if fill_notes:
        print("\nFilled missing reps:")
        for n in fill_notes:
            print(f"  - {n}")

    out = args.out_npy.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, tensor)
    print(f"\nSaved tensor: {out}")
    print(
        f"Shape: {tensor.shape}  (T,R,F,L,C)  pose_scope={args.pose_scope!r}  "
        f"L={num_landmarks}  coord_preset={args.coord_preset!r}  C={C}"
    )
    names = out.with_suffix(".tasks.txt")
    names.write_text("\n".join(t for t, _ in selected) + "\n", encoding="utf-8")
    print(f"Saved task order: {names}")


if __name__ == "__main__":
    main()

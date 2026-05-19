# Extract landmarks from recorded videos → CSV

**Separate from** the live webcam `main.py` — nothing here changes your recording app.

This folder follows the same idea as [kinivi/hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe): **hand (and here also body) keypoints → tabular data**, but exports **CSV** with MediaPipe **Pose + Hands** (Tasks API) instead of their in-app logging keys.

## Install (once)

```bash
cd landmark_export
pip install -r requirements-landmark-export.txt
```

## 1) Full video → landmarks CSV (no window)

Processes every frame and writes one row per frame:

```bash
python extract_landmarks_from_video.py --video /path/to/your/pose_hands_YYYYMMDD_HHMMSS.mp4
```

Output (next to the video unless `--out-dir` is set):

- `pose_hands_YYYYMMDD_HHMMSS_landmarks.csv`

Columns include `frame_index`, `timestamp_ms`, then **pose** (33× x,y,z), **left_hand** (21×), **right_hand** (21×). Missing hands use empty cells.

## 2) Same CSV + activity segments (keyboard shortcuts)

While the video plays, mark **start** / **end** of an activity and a **numeric class label** (0–9):

| Key | Action |
|-----|--------|
| `[` or `b` | **Start** of activity at current frame |
| `]` or `e` | **End** of activity at current frame (writes one row to events file) |
| `0`–`9` | **Class label** used when you press `]` / `e` |
| `q` | Quit and save |

```bash
python extract_landmarks_from_video.py --video /path/to/clip.mp4 --annotate
```

Outputs:

- `*_landmarks.csv` — same per-frame landmarks as mode (1)
- `*_events.csv` — columns: `event_index`, `start_frame`, `end_frame`, `label`

Use `start_frame` / `end_frame` to slice rows in `*_landmarks.csv` for each activity.

## 3) Only the “task performing” window (recommended for many clips)

Recordings often include **idle time** before/after the task. Export **only** the frames where the task happens using **either** frame indices **or** seconds.

### One video

```bash
python extract_landmarks_from_video.py --video clip.mp4 --start-frame 120 --end-frame 480
# or using seconds (uses the file’s FPS):
python extract_landmarks_from_video.py --video clip.mp4 --start-sec 4.0 --end-sec 16.0
```

Output file name includes the window, e.g. `…_win_120_480_landmarks.csv`. Rows are **only** inside that range; `frame_index` is still the **original** index in the file.

### Many videos (e.g. 20 tasks × 3 reps ≈ 60 files)

1. Put all `.mp4` / `.avi` files in one folder (videos can differ in length and resolution — that is OK).
2. Create a spreadsheet **`segments.csv`** (see `example_segments_manifest.csv`) with one row per take:
   - **`video_filename`** — file name (or **`path`** with full path).
   - **`start_frame`**, **`end_frame`** — inclusive task window **or** **`start_sec`**, **`end_sec`**.
   - **`task_id`**, **`trial`** (or **`rep`**) — labels for your study (1–20 and 1–3).
   - Optional **`clip_id`** — any ID you like.

3. Run:

```bash
python extract_landmarks_from_video.py \
  --manifest /path/to/segments.csv \
  --video-dir /path/to/folder/with/videos \
  --out-dir /path/to/landmarks_export
```

Each row produces one CSV like `…_task1_rep1_landmarks.csv` with **task metadata columns** plus **only** frames inside the window.

Optional: merge every clip into **one** training file:

```bash
python extract_landmarks_from_video.py \
  --manifest segments.csv \
  --video-dir ./recordings \
  --out-dir ./out \
  --merge-csv ./out/all_tasks_landmarks.csv
```

## Notes

- Models download once under `~/.mediapipe_pose_hand_models/` (same as many MediaPipe samples).
- Re-run detection on your **recorded** video; coordinates are **re-estimated**, not read from the pixels of the overlay text.

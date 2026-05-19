"""Task catalog and reference-video resolution for the Streamlit capture UI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS_FILE = PROJECT_ROOT / "tasks_ipn20.txt"
RECORDINGS_DIR = PROJECT_ROOT / "recordings"

# Timeline from "Start recording" (matches 15 s reference clip).
SESSION_DURATION_SEC = 15.0    # total session + MP4 length (wall clock from button click)
SESSION_CUE_START_SEC = 5.0    # on-screen START banner (perform task from here)
SESSION_CUE_STOP_SEC = 11.0    # on-screen STOP banner (task ends; recording continues to 15 s)

# Fixed writer FPS so saved MP4 duration = SESSION_DURATION_SEC (not ~1 min).
OUTPUT_FPS = 30.0
MAX_RECORDED_FRAMES = int(SESSION_DURATION_SEC * OUTPUT_FPS)

# Aliases used by capture / UI
SESSION_END_SEC = SESSION_DURATION_SEC
SESSION_RECORD_START_SEC = 0.0          # MP4 starts immediately when session starts
SESSION_RECORD_STOP_SEC = SESSION_DURATION_SEC
RECORD_COUNTDOWN_SEC = SESSION_CUE_START_SEC  # prep overlay 0–5 s while file is recording

LANDMARK_START_SEC = SESSION_CUE_START_SEC
LANDMARK_END_SEC = SESSION_CUE_STOP_SEC

# Landmark CSV: task window inside the 15 s MP4 (seconds in the saved file).
LANDMARK_EXPORT_START_SEC = SESSION_CUE_START_SEC
LANDMARK_EXPORT_END_SEC = SESSION_CUE_STOP_SEC


def normalize_task_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"_landmarks$", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("bursh", "brush")
    if s == "carrying bag":
        s = "carrying the bag"
    if s == "tying":
        s = "tying shoe"
    return s


def task_slug(task_name: str) -> str:
    s = normalize_task_name(task_name)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def sanitize_subject_id(subject_id: str) -> str:
    s = subject_id.strip()
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def subject_recordings_root(subject_id: str) -> Path:
    return PROJECT_ROOT / f"{sanitize_subject_id(subject_id)}_recordings"


def subject_landmarks_root(subject_id: str) -> Path:
    return PROJECT_ROOT / f"{sanitize_subject_id(subject_id)}_landmarks_output"


def recording_path(subject_id: str, task_name: str, timestamp: str) -> Path:
    slug = task_slug(task_name)
    sid = sanitize_subject_id(subject_id)
    folder = subject_recordings_root(subject_id) / slug
    return folder / f"{sid}_{slug}_{timestamp}.mp4"


def landmarks_csv_path(subject_id: str, task_name: str, timestamp: str) -> Path:
    slug = task_slug(task_name)
    sid = sanitize_subject_id(subject_id)
    folder = subject_landmarks_root(subject_id) / slug
    return folder / f"{sid}_{slug}_{timestamp}_landmarks.csv"


def load_canonical_tasks() -> list[str]:
    if not TASKS_FILE.is_file():
        raise FileNotFoundError(f"Task list not found: {TASKS_FILE}")
    tasks: list[str] = []
    for line in TASKS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tasks.append(line)
    if len(tasks) != 20:
        raise ValueError(f"Expected 20 tasks in {TASKS_FILE}, found {len(tasks)}")
    return tasks


def _is_rep1_stem(stem: str) -> bool:
    stem = stem.strip()
    if stem.lower() == "open":
        return False
    return not re.search(r"[23]$", stem)


@dataclass(frozen=True)
class TaskEntry:
    name: str
    slug: str
    reference_video: Path


def build_task_registry(recordings_dir: Path | None = None) -> list[TaskEntry]:
    rec_dir = recordings_dir or RECORDINGS_DIR
    tasks = load_canonical_tasks()

    rep1_by_norm: dict[str, Path] = {}
    for p in sorted(rec_dir.glob("*.mp4")):
        if not _is_rep1_stem(p.stem):
            continue
        key = normalize_task_name(p.stem)
        if key not in rep1_by_norm:
            rep1_by_norm[key] = p

    entries: list[TaskEntry] = []
    missing: list[str] = []
    for name in tasks:
        key = normalize_task_name(name)
        ref = rep1_by_norm.get(key)
        if ref is None:
            missing.append(name)
            continue
        entries.append(TaskEntry(name=name, slug=task_slug(name), reference_video=ref))

    if missing:
        raise FileNotFoundError(
            "Missing rep-1 reference videos for: " + ", ".join(missing)
        )
    return entries


def get_task_by_slug(registry: list[TaskEntry], slug: str) -> TaskEntry | None:
    for entry in registry:
        if entry.slug == slug:
            return entry
    return None

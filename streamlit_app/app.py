"""
Streamlit UI: subject task recording + landmark export.

Run from repo root:
  streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_app.capture import LiveCapture
from streamlit_app.landmarks import export_landmarks_for_recording
from streamlit_app.task_registry import (
    LANDMARK_END_SEC,
    LANDMARK_START_SEC,
    RECORD_COUNTDOWN_SEC,
    SESSION_CUE_START_SEC,
    SESSION_CUE_STOP_SEC,
    SESSION_DURATION_SEC,
    build_task_registry,
    get_task_by_slug,
    landmarks_csv_path,
    recording_path,
    sanitize_subject_id,
    subject_landmarks_root,
    subject_recordings_root,
)

_LIVE_FEED_INTERVAL_SEC = 0.2


def _finish_recording(
    saved: Path,
    *,
    subject_id: str,
    task_name: str,
    timestamp: str,
    status_slot,
    auto: bool = False,
) -> None:
    st.session_state.last_video_path = saved
    prefix = "Auto-stopped — saved" if auto else "Saved"
    status_slot.success(f"{prefix} video: `{saved}`")
    csv_out = landmarks_csv_path(subject_id, task_name, timestamp)
    with st.spinner("Extracting landmarks…"):
        lm_err, lm_warn = export_landmarks_for_recording(saved, csv_out)
    if lm_err:
        st.session_state.last_landmark_error = lm_err
        status_slot.error(f"Landmark export failed: {lm_err}")
    else:
        st.session_state.last_csv_path = csv_out
        st.session_state.last_landmark_error = None
        msg = f"Saved landmarks: `{csv_out}`"
        if lm_warn:
            msg += f" ({lm_warn})"
        status_slot.success(msg)


def _init_state() -> None:
    defaults = {
        "subject_id": "",
        "selected_task_slug": None,
        "capture": None,
        "session_active": False,
        "last_video_path": None,
        "last_csv_path": None,
        "last_landmark_error": None,
        "current_timestamp": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _get_capture() -> LiveCapture:
    cap = st.session_state.capture
    if cap is None:
        cap = LiveCapture()
        st.session_state.capture = cap
    return cap


@st.cache_resource
def _load_registry():
    return build_task_registry()


def _render_output_browser(subject_id: str) -> None:
    sid = sanitize_subject_id(subject_id)
    if not sid:
        st.info("Enter a Subject ID to browse recordings and landmarks.")
        return

    rec_root = subject_recordings_root(subject_id)
    lm_root = subject_landmarks_root(subject_id)

    col_v, col_l = st.columns(2)

    with col_v:
        st.subheader("Recorded videos")
        if not rec_root.is_dir():
            st.caption(f"No folder yet: `{rec_root.name}/`")
        else:
            task_dirs = sorted([d for d in rec_root.iterdir() if d.is_dir()])
            if not task_dirs:
                st.caption("No task subfolders yet.")
            for task_dir in task_dirs:
                videos = sorted(task_dir.glob("*.mp4"))
                with st.expander(f"{task_dir.name} ({len(videos)} video(s))", expanded=False):
                    for vp in videos:
                        st.caption(vp.name)
                        st.video(str(vp))
                        st.download_button(
                            "Download MP4",
                            data=vp.read_bytes(),
                            file_name=vp.name,
                            key=f"dl_vid_{vp}",
                        )

    with col_l:
        st.subheader("Landmark CSVs")
        if not lm_root.is_dir():
            st.caption(f"No folder yet: `{lm_root.name}/`")
        else:
            task_dirs = sorted([d for d in lm_root.iterdir() if d.is_dir()])
            if not task_dirs:
                st.caption("No task subfolders yet.")
            for task_dir in task_dirs:
                csvs = sorted(task_dir.glob("*.csv"))
                with st.expander(f"{task_dir.name} ({len(csvs)} CSV(s))", expanded=False):
                    for cp in csvs:
                        st.caption(cp.name)
                        try:
                            df = pd.read_csv(cp, nrows=8)
                            st.dataframe(df, use_container_width=True)
                        except Exception as exc:
                            st.warning(f"Could not preview: {exc}")
                        st.download_button(
                            "Download CSV",
                            data=cp.read_bytes(),
                            file_name=cp.name,
                            key=f"dl_csv_{cp}",
                        )


def _live_feed_fragment() -> None:
    capture = _get_capture()

    if not st.session_state.session_active:
        st.info(
            f"Press **Start recording** — full **{SESSION_DURATION_SEC:.0f} s** session. "
            f"**START** at **{SESSION_CUE_START_SEC:.0f} s**, **STOP** at "
            f"**{SESSION_CUE_STOP_SEC:.0f} s** (MP4 is the whole {SESSION_DURATION_SEC:.0f} s)."
        )
        return

    frame_rgb = capture.get_latest_frame_rgb()
    if frame_rgb is not None:
        session_t = capture.session_elapsed_sec
        left = capture.session_seconds_left
        if capture.is_countdown:
            label = f"Prep — {session_t:.1f}s / {SESSION_DURATION_SEC:.0f}s"
            st.caption(
                f"**Countdown** — **START** at **{SESSION_CUE_START_SEC:.0f} s**. "
                f"MP4 is recording the full **{SESSION_DURATION_SEC:.0f} s**."
            )
        elif capture.is_recording:
            frames = capture.record_frame_count
            label = f"Session {session_t:.1f}s / {SESSION_DURATION_SEC:.0f}s ({frames} frames)"
            if session_t < SESSION_CUE_START_SEC:
                st.caption("Recording… waiting for **START** cue.")
            elif session_t < SESSION_CUE_STOP_SEC:
                st.caption(
                    f"**Task active** — **STOP** at **{SESSION_CUE_STOP_SEC:.0f} s** "
                    f"({SESSION_CUE_STOP_SEC - session_t:.1f}s left)"
                )
            elif capture.is_post_stop_cue:
                st.caption(
                    f"**STOP** shown — hold still until session ends (**{left:.1f}s** left)"
                )
            else:
                st.caption(f"Finishing session (**{left:.1f}s** left)")
        else:
            label = "Camera"
        st.image(frame_rgb, channels="RGB", caption=label, use_container_width=True)
    else:
        st.warning("Waiting for camera…")

    if capture.error:
        st.error(capture.error)


if hasattr(st, "fragment"):
    _live_feed_fragment = st.fragment(run_every=_LIVE_FEED_INTERVAL_SEC)(
        _live_feed_fragment
    )


def main() -> None:
    st.set_page_config(page_title="Arm tracking capture", layout="wide")
    _init_state()

    st.title("Arm tracking — task capture")
    st.caption(
        f"Session **{SESSION_DURATION_SEC:.0f} s** (matches reference). "
        f"**START** at **{SESSION_CUE_START_SEC:.0f} s**, **STOP** at **{SESSION_CUE_STOP_SEC:.0f} s**. "
        f"Landmarks exported from **{LANDMARK_START_SEC:.0f}–{LANDMARK_END_SEC:.0f} s** in the MP4."
    )

    try:
        registry = _load_registry()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    capture = _get_capture()

    with st.sidebar:
        st.header("Subject")
        subject_input = st.text_input(
            "Subject ID",
            value=st.session_state.subject_id,
            placeholder="e.g. Subject_01",
            disabled=st.session_state.session_active,
        )
        st.session_state.subject_id = subject_input.strip()
        sid = sanitize_subject_id(st.session_state.subject_id)
        if subject_input and not sid:
            st.warning("Subject ID must contain letters, numbers, or underscores.")

        st.divider()
        st.markdown("**Tasks** (20 ADLs)")
        session_on = st.session_state.session_active

        for row_start in range(0, 20, 4):
            cols = st.columns(4)
            for i, col in enumerate(cols):
                idx = row_start + i
                if idx >= len(registry):
                    continue
                entry = registry[idx]
                selected = st.session_state.selected_task_slug == entry.slug
                if col.button(
                    entry.name,
                    key=f"task_{entry.slug}",
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                    disabled=session_on,
                ):
                    st.session_state.selected_task_slug = entry.slug

    selected = None
    if st.session_state.selected_task_slug:
        selected = get_task_by_slug(registry, st.session_state.selected_task_slug)

    if selected is None:
        st.info("Select a task from the sidebar to view the reference video and record.")
    else:
        st.header(selected.name)
        if sid:
            st.markdown(f"Subject: **{sid}**")
        else:
            st.warning("Enter a Subject ID in the sidebar before recording.")

        st.subheader("Reference video (rep 1)")
        st.video(str(selected.reference_video))

        st.markdown(
            f"**Session:** {SESSION_DURATION_SEC:.0f} s &nbsp;|&nbsp; "
            f"**START:** {SESSION_CUE_START_SEC:.0f} s &nbsp;|&nbsp; "
            f"**STOP:** {SESSION_CUE_STOP_SEC:.0f} s &nbsp;|&nbsp; "
            f"**Landmarks:** {LANDMARK_START_SEC:g}–{LANDMARK_END_SEC:g} s in file"
        )

        st.subheader("Webcam")
        _live_feed_fragment()

        status_slot = st.empty()

        # Auto-stop finished in the background thread — finalize here.
        if st.session_state.session_active and not capture.is_running:
            was_auto = capture.auto_stopped
            saved = capture.poll_completed_recording()
            st.session_state.session_active = False
            ts = st.session_state.current_timestamp or datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            if saved is not None and saved.is_file():
                _finish_recording(
                    saved,
                    subject_id=st.session_state.subject_id,
                    task_name=selected.name,
                    timestamp=ts,
                    status_slot=status_slot,
                    auto=was_auto,
                )
            else:
                status_slot.warning(
                    "Session ended. No video saved (stopped during countdown)."
                )
            st.rerun()

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            start_rec = st.button(
                "Start recording",
                disabled=st.session_state.session_active or not sid,
                type="primary",
                use_container_width=True,
            )

        with btn_col2:
            stop_rec = st.button(
                "Stop recording",
                disabled=not st.session_state.session_active,
                use_container_width=True,
            )

        if start_rec and sid:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.current_timestamp = ts
            out_mp4 = recording_path(st.session_state.subject_id, selected.name, ts)
            err = capture.start_session(out_mp4)
            if err:
                status_slot.error(err)
                st.session_state.session_active = False
            else:
                st.session_state.session_active = True
                st.session_state.last_video_path = None
                st.session_state.last_csv_path = None
                st.session_state.last_landmark_error = None
                st.rerun()

        if stop_rec and st.session_state.session_active:
            saved = capture.stop()
            st.session_state.session_active = False
            ts = st.session_state.current_timestamp or datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            if saved is None or not saved.is_file():
                status_slot.warning(
                    "Session ended. No video saved "
                    "(stop during countdown, or no frames were captured)."
                )
            else:
                _finish_recording(
                    saved,
                    subject_id=st.session_state.subject_id,
                    task_name=selected.name,
                    timestamp=ts,
                    status_slot=status_slot,
                    auto=False,
                )

            st.rerun()

        if st.session_state.last_video_path:
            st.markdown(f"Last video: `{st.session_state.last_video_path}`")
        if st.session_state.last_csv_path:
            st.markdown(f"Last CSV: `{st.session_state.last_csv_path}`")
        if st.session_state.last_landmark_error and st.session_state.last_video_path:
            if st.button("Retry landmark export"):
                vp = st.session_state.last_video_path
                ts_retry = st.session_state.current_timestamp
                if not ts_retry:
                    parts = vp.stem.split("_")
                    ts_retry = (
                        "_".join(parts[-2:])
                        if len(parts) >= 3
                        else datetime.now().strftime("%Y%m%d_%H%M%S")
                    )
                csv_out = landmarks_csv_path(
                    st.session_state.subject_id, selected.name, ts_retry
                )
                with st.spinner("Retrying landmark export…"):
                    lm_err, lm_warn = export_landmarks_for_recording(vp, csv_out)
                if lm_err:
                    st.session_state.last_landmark_error = lm_err
                    st.error(lm_err)
                else:
                    st.session_state.last_csv_path = csv_out
                    st.session_state.last_landmark_error = None
                    ok_msg = f"Saved: `{csv_out}`"
                    if lm_warn:
                        ok_msg += f" ({lm_warn})"
                    st.success(ok_msg)
                st.rerun()

    st.divider()
    _render_output_browser(st.session_state.subject_id)


if __name__ == "__main__":
    main()

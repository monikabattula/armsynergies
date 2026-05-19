"""
posehandDetector: Holistic when `mp.solutions` exists, else Pose + Hands (Tasks API).

Fixes:
  AttributeError: module 'mediapipe' has no attribute 'solutions'

Your custom_drawing_utils.draw_landmarks should accept .landmark lists (works with this file).
"""
from __future__ import annotations

import os
import ssl
import time
import urllib.request
from typing import Any, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import custom_drawing_utils as cdu

_USE_LEGACY = hasattr(mp, "solutions")

_POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]

_HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]

_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".mediapipe_pose_hand_models")


def _download_model(url: str, path: str) -> None:
    if os.path.isfile(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {os.path.basename(path)} ...")

    def fetch(ctx: Optional[ssl.SSLContext] = None) -> None:
        opener = (
            urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
            if ctx
            else urllib.request.build_opener()
        )
        with opener.open(url, timeout=120) as r:
            with open(path, "wb") as f:
                f.write(r.read())

    try:
        fetch()
    except Exception as e:
        if "certificate" in str(e).lower() or "ssl" in str(e).lower():
            try:
                import certifi

                fetch(ssl.create_default_context(cafile=certifi.where()))
            except ImportError:
                fetch(ssl._create_unverified_context())
        else:
            raise


def _ensure_task_models() -> Tuple[str, str]:
    pose_url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )
    hand_url = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    pp = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")
    hp = os.path.join(_MODEL_DIR, "hand_landmarker.task")
    _download_model(pose_url, pp)
    _download_model(hand_url, hp)
    return pp, hp


def _wrap_landmarks(pts: List[Any]) -> Any:
    """Tasks landmarks → object with .landmark and HasField (for draw_landmarks)."""

    class _P:
        __slots__ = ("x", "y", "z")

        def __init__(self, lm: Any) -> None:
            self.x = float(lm.x)
            self.y = float(lm.y)
            self.z = float(getattr(lm, "z", 0.0) or 0.0)

        def HasField(self, _name: str) -> bool:
            return False

    return type("NL", (), {"landmark": [_P(lm) for lm in pts]})()


class _TaskMeta:
    """Same indices as MediaPipe Holistic enums (pose 33, hand 21)."""

    class PoseLandmark:
        LEFT_SHOULDER = 11
        LEFT_ELBOW = 13
        LEFT_WRIST = 15
        LEFT_HIP = 23
        RIGHT_SHOULDER = 12
        RIGHT_ELBOW = 14
        RIGHT_WRIST = 16
        RIGHT_HIP = 24

    class HandLandmark:
        WRIST = 0
        THUMB_CMC = 1
        THUMB_MCP = 2
        THUMB_IP = 3
        THUMB_TIP = 4
        INDEX_FINGER_MCP = 5
        INDEX_FINGER_PIP = 6
        INDEX_FINGER_DIP = 7
        INDEX_FINGER_TIP = 8
        MIDDLE_FINGER_MCP = 9
        MIDDLE_FINGER_PIP = 10
        MIDDLE_FINGER_DIP = 11
        MIDDLE_FINGER_TIP = 12
        RING_FINGER_MCP = 13
        RING_FINGER_PIP = 14
        RING_FINGER_DIP = 15
        RING_FINGER_TIP = 16
        PINKY_MCP = 17
        PINKY_PIP = 18
        PINKY_DIP = 19
        PINKY_TIP = 20

    POSE_CONNECTIONS = _POSE_CONNECTIONS
    HAND_CONNECTIONS = _HAND_CONNECTIONS


class posehandDetector:
    def __init__(
        self,
        customDraw,
        modelComplexity=2,
        smooth_landmarks=True,
        static_image_mode=False,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.modelComplexity = modelComplexity
        self.smoothLandmarks = smooth_landmarks
        self.staticMode = static_image_mode
        self.segmentation = enable_segmentation
        self.minDetectionCon = min_detection_confidence
        self.minTrackCon = min_tracking_confidence
        self.mpCustDraw = customDraw

        self._legacy = _USE_LEGACY
        self.results: Any = None
        self._video_t0: Optional[float] = None  # monotonic start for Tasks VIDEO timestamps

        if self._legacy:
            self.mpWhole = mp.solutions.holistic
            # Same constructor as your original snippet (0.8 / 0.8).
            self.whole = self.mpWhole.Holistic(
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8,
            )
            self.mpDraw = mp.solutions.drawing_utils
            self.mpDrawStyle = mp.solutions.drawing_styles
            self._pose_lm = None
            self._hands_lm = None
        else:
            from mediapipe.tasks import python as mp_tasks_py
            from mediapipe.tasks.python import vision as mp_vision

            self.mpWhole = _TaskMeta
            self.mpDrawStyle = None
            self.mpDraw = None
            pose_p, hand_p = _ensure_task_models()
            BaseOptions = mp_tasks_py.BaseOptions
            RunningMode = getattr(mp_vision, "VisionRunningMode", None) or getattr(
                mp_vision, "RunningMode", None
            )
            if RunningMode is None:
                from mediapipe.tasks.python.vision.core import vision_task_running_mode as _vrm

                RunningMode = getattr(_vrm, "VisionRunningMode", None) or getattr(
                    _vrm, "RunningMode", None
                )
            if RunningMode is None:
                raise RuntimeError("Could not find MediaPipe VisionRunningMode")

            self._pose_lm = mp_vision.PoseLandmarker.create_from_options(
                mp_vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=pose_p),
                    running_mode=RunningMode.VIDEO,
                    min_pose_detection_confidence=min_detection_confidence,
                    min_pose_presence_confidence=min_tracking_confidence,
                )
            )
            self._hands_lm = mp_vision.HandLandmarker.create_from_options(
                mp_vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=hand_p),
                    running_mode=RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_tracking_confidence,
                )
            )
            self.whole = self  # legacy main.py may reference .whole.close()

    def close(self) -> None:
        if self._legacy:
            self.whole.close()
        else:
            if self._pose_lm is not None:
                self._pose_lm.close()
            if self._hands_lm is not None:
                self._hands_lm.close()

    def _process_tasks(self, img: np.ndarray) -> None:
        assert self._pose_lm is not None and self._hands_lm is not None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode needs monotonic ms (wall clock), not a fixed +33 step.
        if self._video_t0 is None:
            self._video_t0 = time.monotonic()
        ts_ms = int((time.monotonic() - self._video_t0) * 1000)
        pr = self._pose_lm.detect_for_video(mp_image, ts_ms)
        hr = self._hands_lm.detect_for_video(mp_image, ts_ms)

        self.results = type("R", (), {})()
        self.results.pose_landmarks = None
        self.results.left_hand_landmarks = None
        self.results.right_hand_landmarks = None

        if pr.pose_landmarks:
            self.results.pose_landmarks = _wrap_landmarks(pr.pose_landmarks[0])

        if hr.hand_landmarks and hr.handedness:
            for hand_lms, cats in zip(hr.hand_landmarks, hr.handedness):
                cat = cats[0] if cats else None
                name = (
                    getattr(cat, "category_name", None)
                    or getattr(cat, "display_name", None)
                    or "Right"
                )
                wrapped = _wrap_landmarks(hand_lms)
                if name == "Left":
                    self.results.left_hand_landmarks = wrapped
                else:
                    self.results.right_hand_landmarks = wrapped

    def findWhole(self, img: np.ndarray) -> np.ndarray:
        if self._legacy:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.whole.process(imgRGB)
            if self.results.pose_landmarks:
                self.mpCustDraw(
                    img,
                    self.results.pose_landmarks,
                    self.mpWhole.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mpDrawStyle.get_default_pose_landmarks_style(),
                )
            if self.results.left_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.left_hand_landmarks,
                    self.mpWhole.HAND_CONNECTIONS,
                )
            if self.results.right_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.right_hand_landmarks,
                    self.mpWhole.HAND_CONNECTIONS,
                )
            return img

        self._process_tasks(img)
        if self.results.pose_landmarks:
            self.mpCustDraw(
                img,
                self.results.pose_landmarks,
                _POSE_CONNECTIONS,
                landmark_drawing_spec=cdu.DrawingSpec(color=cdu.RED_COLOR),
            )
        if self.results.left_hand_landmarks:
            cdu.draw_landmarks(img, self.results.left_hand_landmarks, _HAND_CONNECTIONS)
        if self.results.right_hand_landmarks:
            cdu.draw_landmarks(img, self.results.right_hand_landmarks, _HAND_CONNECTIONS)
        return img

    def findPosition(self, img: np.ndarray) -> list:
        if self._legacy:
            if self.results is None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.results = self.whole.process(img_rgb)
        elif self.results is None:
            self._process_tasks(img)
        left = self.results.left_hand_landmarks
        right = self.results.right_hand_landmarks
        l, r = "LEFT", "RIGHT"
        left_landmark = self.handPosition([], left, l, img)
        right_landmark = self.handPosition([], right, r, img)
        pose_landmark = self.bodyPosition([], img)
        return [pose_landmark, left_landmark, right_landmark]

    def bodyPosition(self, landmark, img: np.ndarray) -> list:
        landmarkList: list = []
        if self.results and self.results.pose_landmarks:
            h, w, c = img.shape
            plm = self.results.pose_landmarks.landmark
            P = self.mpWhole.PoseLandmark
            left_shoulder = [
                int(plm[P.LEFT_SHOULDER].x * w),
                int(plm[P.LEFT_SHOULDER].y * h),
                plm[P.LEFT_SHOULDER].z,
            ]
            left_elbow = [
                int(plm[P.LEFT_ELBOW].x * w),
                int(plm[P.LEFT_ELBOW].y * h),
                plm[P.LEFT_ELBOW].z,
            ]
            left_wrist = [
                int(plm[P.LEFT_WRIST].x * w),
                int(plm[P.LEFT_WRIST].y * h),
                plm[P.LEFT_WRIST].z,
            ]
            left_hip = [
                int(plm[P.LEFT_HIP].x * w),
                int(plm[P.LEFT_HIP].y * h),
                plm[P.LEFT_HIP].z,
            ]
            right_shoulder = [
                int(plm[P.RIGHT_SHOULDER].x * w),
                int(plm[P.RIGHT_SHOULDER].y * h),
                plm[P.RIGHT_SHOULDER].z,
            ]
            right_elbow = [
                int(plm[P.RIGHT_ELBOW].x * w),
                int(plm[P.RIGHT_ELBOW].y * h),
                plm[P.RIGHT_ELBOW].z,
            ]
            right_wrist = [
                int(plm[P.RIGHT_WRIST].x * w),
                int(plm[P.RIGHT_WRIST].y * h),
                plm[P.RIGHT_WRIST].z,
            ]
            right_hip = [
                int(plm[P.RIGHT_HIP].x * w),
                int(plm[P.RIGHT_HIP].y * h),
                plm[P.RIGHT_HIP].z,
            ]
            left = [[11, left_shoulder], [13, left_elbow], [15, left_wrist], [23, left_hip]]
            right = [[12, right_shoulder], [14, right_elbow], [16, right_wrist], [24, right_hip]]
            landmarkList = left + right
        return landmarkList

    def handPosition(self, landmark, hand, handText, img: np.ndarray) -> list:
        landmark = []
        if hand:
            h, w, c = img.shape
            H = self.mpWhole.HandLandmark
            wrist = [
                int(hand.landmark[H.WRIST].x * w),
                int(hand.landmark[H.WRIST].y * h),
            ]
            thumb_cmc = [
                int(hand.landmark[H.THUMB_CMC].x * w),
                int(hand.landmark[H.THUMB_CMC].y * h),
                hand.landmark[H.THUMB_CMC].z,
            ]
            thumb_mcp = [
                int(hand.landmark[H.THUMB_MCP].x * w),
                int(hand.landmark[H.THUMB_MCP].y * h),
                hand.landmark[H.THUMB_MCP].z,
            ]
            thumb_ip = [
                int(hand.landmark[H.THUMB_IP].x * w),
                int(hand.landmark[H.THUMB_IP].y * h),
                hand.landmark[H.THUMB_IP].z,
            ]
            thumb_tip = [
                int(hand.landmark[H.THUMB_TIP].x * w),
                int(hand.landmark[H.THUMB_TIP].y * h),
                hand.landmark[H.THUMB_TIP].z,
            ]
            index_mcp = [
                int(hand.landmark[H.INDEX_FINGER_MCP].x * w),
                int(hand.landmark[H.INDEX_FINGER_MCP].y * h),
            ]
            index_pip = [
                int(hand.landmark[H.INDEX_FINGER_PIP].x * w),
                int(hand.landmark[H.INDEX_FINGER_PIP].y * h),
            ]
            index_dip = [
                int(hand.landmark[H.INDEX_FINGER_DIP].x * w),
                int(hand.landmark[H.INDEX_FINGER_DIP].y * h),
            ]
            index_tip = [
                int(hand.landmark[H.INDEX_FINGER_TIP].x * w),
                int(hand.landmark[H.INDEX_FINGER_TIP].y * h),
            ]
            middle_mcp = [
                int(hand.landmark[H.MIDDLE_FINGER_MCP].x * w),
                int(hand.landmark[H.MIDDLE_FINGER_MCP].y * h),
            ]
            middle_pip = [
                int(hand.landmark[H.MIDDLE_FINGER_PIP].x * w),
                int(hand.landmark[H.MIDDLE_FINGER_PIP].y * h),
            ]
            middle_dip = [
                int(hand.landmark[H.MIDDLE_FINGER_DIP].x * w),
                int(hand.landmark[H.MIDDLE_FINGER_DIP].y * h),
            ]
            middle_tip = [
                int(hand.landmark[H.MIDDLE_FINGER_TIP].x * w),
                int(hand.landmark[H.MIDDLE_FINGER_TIP].y * h),
            ]
            ring_mcp = [
                int(hand.landmark[H.RING_FINGER_MCP].x * w),
                int(hand.landmark[H.RING_FINGER_MCP].y * h),
            ]
            ring_pip = [
                int(hand.landmark[H.RING_FINGER_PIP].x * w),
                int(hand.landmark[H.RING_FINGER_PIP].y * h),
            ]
            ring_dip = [
                int(hand.landmark[H.RING_FINGER_DIP].x * w),
                int(hand.landmark[H.RING_FINGER_DIP].y * h),
            ]
            ring_tip = [
                int(hand.landmark[H.RING_FINGER_TIP].x * w),
                int(hand.landmark[H.RING_FINGER_TIP].y * h),
            ]
            pinky_mcp = [
                int(hand.landmark[H.PINKY_MCP].x * w),
                int(hand.landmark[H.PINKY_MCP].y * h),
            ]
            pinky_pip = [
                int(hand.landmark[H.PINKY_PIP].x * w),
                int(hand.landmark[H.PINKY_PIP].y * h),
            ]
            pinky_dip = [
                int(hand.landmark[H.PINKY_DIP].x * w),
                int(hand.landmark[H.PINKY_DIP].y * h),
            ]
            pinky_tip = [
                int(hand.landmark[H.PINKY_TIP].x * w),
                int(hand.landmark[H.PINKY_TIP].y * h),
            ]
            landmark = [
                [0, handText, wrist],
                [1, handText, thumb_cmc],
                [2, handText, thumb_mcp],
                [3, handText, thumb_ip],
                [4, handText, thumb_tip],
                [5, handText, index_mcp],
                [6, handText, index_pip],
                [7, handText, index_dip],
                [8, handText, index_tip],
                [9, handText, middle_mcp],
                [10, handText, middle_pip],
                [11, handText, middle_dip],
                [12, handText, middle_tip],
                [13, handText, ring_mcp],
                [14, handText, ring_pip],
                [15, handText, ring_dip],
                [16, handText, ring_tip],
                [17, handText, pinky_mcp],
                [18, handText, pinky_pip],
                [19, handText, pinky_dip],
                [20, handText, pinky_tip],
            ]
        return landmark

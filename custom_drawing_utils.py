# Copyright 2020 The MediaPipe Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Drop-in replacement: same behavior as mediapipe.solutions.drawing_utils for
# draw_landmarks / draw_detection / plot_landmarks, but WITHOUT
# `from mediapipe.framework.formats import ...` (that module is missing on
# current PyPI mediapipe "tasks" wheels).

"""MediaPipe solution drawing utils (protobuf types replaced with Any)."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, List, Mapping, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

_RELATIVE_BOUNDING_BOX_FORMAT = 1

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)


@dataclasses.dataclass
class DrawingSpec:
    color: Tuple[int, int, int] = WHITE_COLOR
    thickness: int = 2
    circle_radius: int = 2


def _normalized_to_pixel_coordinates(
    normalized_x: float,
    normalized_y: float,
    image_width: int,
    image_height: int,
) -> Union[None, Tuple[int, int]]:
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def draw_detection(
    image: np.ndarray,
    detection: Any,
    keypoint_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR),
    bbox_drawing_spec: DrawingSpec = DrawingSpec(),
) -> None:
    if not detection.location_data:
        return
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError("Input image must contain three channel bgr data.")
    image_rows, image_cols, _ = image.shape
    location = detection.location_data
    if location.format != _RELATIVE_BOUNDING_BOX_FORMAT:
        raise ValueError(
            "LocationData must be relative for this drawing funtion to work."
        )
    for keypoint in location.relative_keypoints:
        keypoint_px = _normalized_to_pixel_coordinates(
            keypoint.x, keypoint.y, image_cols, image_rows
        )
        cv2.circle(
            image,
            keypoint_px,
            keypoint_drawing_spec.circle_radius,
            keypoint_drawing_spec.color,
            keypoint_drawing_spec.thickness,
        )
    if not location.HasField("relative_bounding_box"):
        return
    rb = location.relative_bounding_box
    rect_start_point = _normalized_to_pixel_coordinates(
        rb.xmin, rb.ymin, image_cols, image_rows
    )
    rect_end_point = _normalized_to_pixel_coordinates(
        rb.xmin + rb.width, rb.ymin + rb.height, image_cols, image_rows
    )
    cv2.rectangle(
        image,
        rect_start_point,
        rect_end_point,
        bbox_drawing_spec.color,
        bbox_drawing_spec.thickness,
    )


def draw_landmarks(
    image: np.ndarray,
    landmark_list: Any,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[DrawingSpec, Mapping[int, DrawingSpec]] = DrawingSpec(
        color=RED_COLOR
    ),
    connection_drawing_spec: Union[
        DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]
    ] = DrawingSpec(),
) -> None:
    if not landmark_list:
        return
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError("Input image must contain three channel bgr data.")
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    # MediaPipe *pose* has 33 landmarks; *hand* has 21. Face/torso skip list must
    # ONLY apply to pose — on hands it would wrongly drop wrist (0) and fingers.
    num_lm = len(landmark_list.landmark)
    is_pose = num_lm == 33
    skip_pose_indices = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        17,
        18,
        19,
        20,
        21,
        22,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
    }
    for idx, landmark in enumerate(landmark_list.landmark):
        if is_pose and idx in skip_pose_indices:
            continue
        if (landmark.HasField("visibility") and landmark.visibility < _VISIBILITY_THRESHOLD) or (
            landmark.HasField("presence") and landmark.presence < _PRESENCE_THRESHOLD
        ):
            continue
        landmark_px = _normalized_to_pixel_coordinates(
            landmark.x, landmark.y, image_cols, image_rows
        )
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    if connections:
        num_landmarks = len(landmark_list.landmark)
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = (
                    connection_drawing_spec[connection]
                    if isinstance(connection_drawing_spec, Mapping)
                    else connection_drawing_spec
                )
                cv2.line(
                    image,
                    idx_to_coordinates[start_idx],
                    idx_to_coordinates[end_idx],
                    drawing_spec.color,
                    drawing_spec.thickness,
                )
    if landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = (
                landmark_drawing_spec[idx]
                if isinstance(landmark_drawing_spec, Mapping)
                else landmark_drawing_spec
            )
            circle_border_radius = max(
                drawing_spec.circle_radius + 1,
                int(drawing_spec.circle_radius * 1.2),
            )
            cv2.circle(
                image,
                landmark_px,
                circle_border_radius,
                WHITE_COLOR,
                drawing_spec.thickness,
            )
            cv2.circle(
                image,
                landmark_px,
                drawing_spec.circle_radius,
                drawing_spec.color,
                drawing_spec.thickness,
            )


def draw_axis(
    image: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    focal_length: Tuple[float, float] = (1.0, 1.0),
    principal_point: Tuple[float, float] = (0.0, 0.0),
    axis_length: float = 0.1,
    axis_drawing_spec: DrawingSpec = DrawingSpec(),
) -> None:
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError("Input image must contain three channel bgr data.")
    image_rows, image_cols, _ = image.shape
    axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axis_cam = np.matmul(rotation, axis_length * axis_world.T).T + translation
    x = axis_cam[..., 0]
    y = axis_cam[..., 1]
    z = axis_cam[..., 2]
    fx, fy = focal_length
    px, py = principal_point
    x_ndc = np.clip(-fx * x / (z + 1e-5) + px, -1.0, 1.0)
    y_ndc = np.clip(-fy * y / (z + 1e-5) + py, -1.0, 1.0)
    x_im = np.int32((1 + x_ndc) * 0.5 * image_cols)
    y_im = np.int32((1 - y_ndc) * 0.5 * image_rows)
    origin = (x_im[0], y_im[0])
    cv2.arrowedLine(image, origin, (x_im[1], y_im[1]), RED_COLOR, axis_drawing_spec.thickness)
    cv2.arrowedLine(image, origin, (x_im[2], y_im[2]), GREEN_COLOR, axis_drawing_spec.thickness)
    cv2.arrowedLine(image, origin, (x_im[3], y_im[3]), BLUE_COLOR, axis_drawing_spec.thickness)


def _normalize_color(color: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return tuple(v / 255.0 for v in color)


def plot_landmarks(
    landmark_list: Any,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR, thickness=5),
    connection_drawing_spec: DrawingSpec = DrawingSpec(color=BLACK_COLOR, thickness=5),
    elevation: int = 10,
    azimuth: int = 10,
) -> None:
    if not landmark_list:
        return
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if (landmark.HasField("visibility") and landmark.visibility < _VISIBILITY_THRESHOLD) or (
            landmark.HasField("presence") and landmark.presence < _PRESENCE_THRESHOLD
        ):
            continue
        ax.scatter3D(
            xs=[-landmark.z],
            ys=[landmark.x],
            zs=[-landmark.y],
            color=_normalize_color(landmark_drawing_spec.color[::-1]),
            linewidth=landmark_drawing_spec.thickness,
        )
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
    if connections:
        num_landmarks = len(landmark_list.landmark)
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx],
                    plotted_landmarks[end_idx],
                ]
                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    color=_normalize_color(connection_drawing_spec.color[::-1]),
                    linewidth=connection_drawing_spec.thickness,
                )
    plt.show()

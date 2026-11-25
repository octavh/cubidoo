"""Cube face detection with a trained YOLO pose model.

The model finds every cube face in the image and, for each one, the 9 sticker
centers as keypoints. Train it with `python train_detector.py`.
"""
from pathlib import Path

import cv2
import numpy as np

from face_detector import Face, MIN_VALID_PATCHES, sample_patch

MODEL_PATH = Path(__file__).parent / "models" / "cube_pose.pt"
MIN_CONFIDENCE = 0.5   # face detections below this are ignored
IMG_SIZE = 480          # must match --imgsz used in train_detector.py


def canonical_order(points):
    """Reorder 9 sticker centers (any row-major order of the face) so that row 0 is the
    top row and column 0 the left column in the image, same as the classic detector:
    of the 4 possible rotations, pick the one whose 'right' direction is closest to
    the image x axis."""
    grid = np.asarray(points, np.float32).reshape(3, 3, 2)

    def tilt(g):
        right = g[1, 2] - g[1, 0]
        return abs(np.arctan2(right[1], right[0]))

    best = min((np.rot90(grid, k) for k in range(4)), key=tilt)
    return best.reshape(9, 2)


def box_contour(box):
    x0, y0, x1, y1 = box
    return np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], np.int32)


class YoloFaceDetector:
    def __init__(self, model_path=MODEL_PATH):
        from ultralytics import YOLO  # imported here so the classic detector works without it
        self.model = YOLO(str(model_path))

    def detect(self, frame, color_detector):
        """Returns (Face or None, candidates) like face_detector.detect_face.
        Candidates are all detected faces, as (center, side, angle, contour) tuples."""
        result = self.model.predict(frame, imgsz=IMG_SIZE, conf=MIN_CONFIDENCE, verbose=False)[0]
        if result.keypoints is None or len(result.boxes) == 0:
            return None, []

        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        candidates = [(None, None, None, box_contour(b)) for b in boxes]

        # The face being shown to the camera is the biggest one.
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        points = canonical_order(keypoints[int(np.argmax(areas))])
        grid = points.reshape(3, 3, 2)
        spacing = float(np.mean(np.concatenate([
            np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
            np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
        ])))

        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        patches = [sample_patch(hsv, p, spacing) for p in points]
        if sum(color_detector.is_sticker(p) for p in patches) < MIN_VALID_PATCHES:
            return None, candidates
        colors = [color_detector.classify(p) for p in patches]
        return Face(points, colors, patches, spacing), candidates

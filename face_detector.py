from dataclasses import dataclass

import cv2
import numpy as np

MIN_STICKER_AREA = 0.0004  # sticker area bounds, as fraction of frame area
MAX_STICKER_AREA = 0.04
MAX_ASPECT = 1.4           # sticker bounding box width/height ratio
MIN_FILL = 0.75            # contour area / bounding box area
GRID_TOLERANCE = 0.35      # max lattice position error, as fraction of spacing
MIN_MATCHED = 8            # stickers (of 9) that must be found to accept a face
PATCH_FRACTION = 0.35      # sampled patch side as fraction of sticker spacing

GRID_OFFSETS = [(i, j) for j in (-1, 0, 1) for i in (-1, 0, 1)]  # row-major


@dataclass
class Face:
    points: np.ndarray   # 9 sticker centers, row-major, in image coordinates
    colors: list         # 9 color names, row-major
    patches: list        # 9 HSV patches used for classification
    spacing: float       # distance between neighboring sticker centers


def find_square_candidates(frame):
    """Return list of (center xy, side, angle_deg, contour) for square-ish blobs."""
    h, w = frame.shape[:2]
    frame_area = h * w
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 20, 60)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_STICKER_AREA * frame_area < area < MAX_STICKER_AREA * frame_area):
            continue
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(cnt)
        if min(rw, rh) == 0 or max(rw, rh) / min(rw, rh) > MAX_ASPECT:
            continue
        if area / (rw * rh) < MIN_FILL:
            continue
        angle = (angle + 45) % 90 - 45  # normalize to [-45, 45)
        candidates.append((np.array([cx, cy]), np.sqrt(rw * rh), angle, approx))

    # Dilated edges produce nested duplicates per sticker; keep one per location.
    candidates.sort(key=lambda c: c[1])
    unique = []
    for cand in candidates:
        if all(np.linalg.norm(cand[0] - u[0]) > 0.5 * u[1] for u in unique):
            unique.append(cand)
    return unique


def fit_grid(candidates):
    """Find the best 3x3 lattice. Returns (9 image points row-major, spacing) or None."""
    if len(candidates) < MIN_MATCHED:
        return None
    centers = np.array([c[0] for c in candidates])
    sides = np.array([c[1] for c in candidates])

    best = None
    for center, side, angle, _ in candidates:
        similar = (sides > 0.6 * side) & (sides < 1.6 * side)
        dists = np.linalg.norm(centers - center, axis=1)
        neighbor = similar & (dists > 0.9 * side) & (dists < 1.8 * side)
        if not neighbor.any():
            continue
        spacing = dists[neighbor].min()
        t = np.deg2rad(angle)
        u = np.array([np.cos(t), np.sin(t)]) * spacing   # roughly "right"
        v = np.array([-np.sin(t), np.cos(t)]) * spacing  # roughly "down"

        for _ in range(2):  # match, then refine lattice with an affine fit
            predicted = np.array([center + i * u + j * v for i, j in GRID_OFFSETS])
            matched, used = {}, set()
            for k, p in enumerate(predicted):
                d = np.linalg.norm(centers - p, axis=1)
                d[~similar] = np.inf
                m = int(np.argmin(d))
                if d[m] < GRID_TOLERANCE * spacing and m not in used:
                    matched[k] = m
                    used.add(m)
            if len(matched) < MIN_MATCHED:
                break
            # Least-squares affine map from grid coords to image coords.
            A = np.array([[GRID_OFFSETS[k][0], GRID_OFFSETS[k][1], 1] for k in matched])
            B = centers[list(matched.values())]
            sol, *_ = np.linalg.lstsq(A, B, rcond=None)
            u, v, center = sol[0], sol[1], sol[2]

        if len(matched) < MIN_MATCHED:
            continue
        points = np.array([centers[matched[k]] if k in matched else predicted[k] for k in range(9)])
        error = np.mean([np.linalg.norm(centers[m] - predicted[k]) for k, m in matched.items()]) / spacing
        score = (len(matched), -error)
        if best is None or score > best[0]:
            best = (score, points, spacing)

    return None if best is None else (best[1], best[2])


def sample_patch(hsv, point, spacing):
    h, w = hsv.shape[:2]
    half = max(2, int(spacing * PATCH_FRACTION / 2))
    x, y = int(point[0]), int(point[1])
    return hsv[max(0, y - half):min(h, y + half), max(0, x - half):min(w, x + half)]


def detect_face(frame, color_detector):
    """Returns (Face or None, square candidates) for a BGR frame."""
    candidates = find_square_candidates(frame)
    grid = fit_grid(candidates)
    if grid is None:
        return None, candidates
    points, spacing = grid
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    patches = [sample_patch(hsv, p, spacing) for p in points]
    if not all(color_detector.is_sticker(p) for p in patches):
        return None, candidates
    colors = [color_detector.classify(p) for p in patches]
    return Face(points, colors, patches, spacing), candidates

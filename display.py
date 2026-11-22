import cv2
import numpy as np

DISPLAY_BGR = {
    "white": (255, 255, 255),
    "yellow": (0, 220, 255),
    "red": (0, 0, 220),
    "orange": (0, 130, 255),
    "green": (0, 180, 0),
    "blue": (220, 80, 0),
    None: (128, 128, 128),
}


def draw_detection(frame, face, candidates):
    for c in candidates:
        cv2.drawContours(frame, [c[3]], -1, (90, 90, 90), 1)
    if face is None:
        return
    points = face.points
    corners = np.array([points[0], points[2], points[8], points[6]])
    center = points[4]
    corners = center + (corners - center) * 1.5  # expand from sticker centers to face edges
    cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 2)
    r = max(4, int(face.spacing * 0.2))
    for p, name in zip(points, face.colors):
        xy = tuple(int(x) for x in p)
        cv2.circle(frame, xy, r, DISPLAY_BGR[name], -1)
        cv2.circle(frame, xy, r, (0, 0, 0), 2)


def render_grid(colors, detected, size=360):
    """colors: list of 9 color names (or None for never-seen), row-major."""
    img = np.full((size + 50, size, 3), 40, np.uint8)
    cell = size // 3
    pad = 6
    for k, name in enumerate(colors):
        r, c = divmod(k, 3)
        x, y = c * cell, r * cell
        cv2.rectangle(img, (x + pad, y + pad), (x + cell - pad, y + cell - pad), DISPLAY_BGR[name], -1)
        cv2.rectangle(img, (x + pad, y + pad), (x + cell - pad, y + cell - pad), (0, 0, 0), 2)
    msg, color = ("Face detected", (120, 255, 120)) if detected else ("No face - showing last", (160, 160, 160))
    cv2.putText(img, msg, (12, size + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return img

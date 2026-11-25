"""Generate synthetic training images for the YOLO cube face detector.

Each image shows a randomly scrambled cube in a random pose, with random lighting,
colors, background, blur, glare and finger-like occluders. Labels are exact and free:
for every face turned toward the camera, its bounding box and its 9 sticker centers
(YOLO pose format, class 0 = face).

    python synth_data.py --train 6000 --val 600
    python synth_data.py --preview 12      # write a few labelled images to look at

Put real photos without cubes (your room, desk, ...) in backgrounds/ to make the
backgrounds more realistic; they are used for about half of the images.
"""
import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np

from virtual_cube import FACE_UP, Cube, random_scramble
from yolo_detector import canonical_order

ROOT = Path(__file__).parent
BACKGROUND_DIR = ROOT / "backgrounds"

BASE_BGR = {
    "white": (235, 235, 235),
    "yellow": (0, 210, 250),
    "red": (40, 20, 200),
    "orange": (0, 110, 255),
    "green": (70, 170, 0),
    "blue": (190, 80, 0),
}
MIN_FRONTALITY = 0.35          # only faces turned at least this much toward the camera get labels
MIN_VISIBLE_KEYPOINTS = 6
FLIP_IDX = [2, 1, 0, 5, 4, 3, 8, 7, 6]  # keypoint order after a horizontal flip
NEGATIVE_FRACTION = 0.08       # images without any cube


# ---------------------------------------------------------------- geometry

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def random_rotation():
    q = np.random.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def sample_pose():
    """World -> camera rotation. Camera coords: x right, y down, z forward."""
    if random.random() < 0.25:
        return random_rotation()
    # Mostly: one face shown to the camera, tilted a bit, held at any roll angle.
    n = np.array(random.choice(list(FACE_UP)))
    up = np.array(FACE_UP[tuple(n)])
    face_to_camera = np.array([np.cross(up, n), -up, -n])  # rows: maps n -> -z, up -> -y
    tilt = rot_x(math.radians(random.uniform(-35, 35))) @ rot_y(math.radians(random.uniform(-35, 35)))
    return tilt @ rot_z(random.uniform(0, 2 * math.pi)) @ face_to_camera


# ---------------------------------------------------------------- drawing

def fill(img, pts, color):
    pts = np.round(np.asarray(pts) * 16).astype(np.int32)
    cv2.fillPoly(img, [pts], tuple(float(c) for c in color), cv2.LINE_AA, shift=4)


def random_palette():
    """Per-cube sticker colors: each cube brand / camera renders them a bit differently."""
    palette = {}
    for name, bgr in BASE_BGR.items():
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0].astype(int)
        hsv[0] = (hsv[0] + random.randint(-6, 6)) % 180
        hsv[1] = np.clip(hsv[1] + random.randint(-50, 20), 0, 255) if name != "white" else random.randint(0, 40)
        hsv[2] = np.clip(hsv[2] + random.randint(-40, 15), 0, 255)
        palette[name] = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0].astype(float)
    return palette


def random_background(h, w, bg_files):
    r = random.random()
    if bg_files and r < 0.5:
        img = cv2.imread(str(random.choice(bg_files)))
        if img is not None:
            scale = max(h / img.shape[0], w / img.shape[1]) * random.uniform(1.0, 1.6)
            img = cv2.resize(img, None, fx=scale, fy=scale)
            y = random.randint(0, img.shape[0] - h)
            x = random.randint(0, img.shape[1] - w)
            return img[y:y + h, x:x + w].copy()
    # Smooth color blobs plus random shapes
    small = np.random.randint(0, 255, (random.randint(2, 6), random.randint(2, 6), 3), np.uint8)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    img = (img * random.uniform(0.3, 1.0) + random.uniform(0, 120)).clip(0, 255).astype(np.uint8)
    for _ in range(random.randint(0, 25)):
        color = [random.randint(0, 255) for _ in range(3)]
        x, y = random.randint(0, w), random.randint(0, h)
        s = random.randint(5, max(6, w // 4))
        kind = random.random()
        if kind < 0.4:
            cv2.rectangle(img, (x, y), (x + s, y + random.randint(5, s)), color, -1)
        elif kind < 0.7:
            cv2.circle(img, (x, y), s // 2, color, -1)
        else:
            cv2.line(img, (x, y), (random.randint(0, w), random.randint(0, h)), color, random.randint(1, 6))
    if random.random() < 0.3:
        draw_distractor_grid(img)
    # Real rooms are much less colorful: blend toward gray by a random amount.
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    keep = random.uniform(0.1, 0.8)
    img = cv2.addWeighted(img, keep, gray, 1 - keep, 0)
    return cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 4))


def draw_distractor_grid(img):
    """Grid of similar squares (keyboard, tiles, ...): teaches the model that not every grid is a cube."""
    h, w = img.shape[:2]
    rows, cols = random.randint(2, 6), random.randint(3, 10)
    s = random.randint(15, 60)
    gap = random.randint(2, s // 3)
    x0, y0 = random.randint(0, w - 1), random.randint(0, h - 1)
    base = np.array([random.randint(0, 255)] * 3) if random.random() < 0.7 else np.random.randint(0, 255, 3)
    for r in range(rows):
        for c in range(cols):
            color = (base + np.random.randint(-15, 15, 3)).clip(0, 255).tolist()
            x, y = x0 + c * (s + gap), y0 + r * (s + gap)
            cv2.rectangle(img, (x, y), (x + s, y + s), color, -1)


def draw_glare(img, center, size):
    overlay = img.copy()
    axes = (max(1, int(size * random.uniform(0.2, 0.6))), max(1, int(size * random.uniform(0.1, 0.3))))
    cv2.ellipse(overlay, tuple(int(v) for v in center), axes, random.uniform(0, 180), 0, 360, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), max(1, size * 0.1))
    alpha = random.uniform(0.3, 0.8)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def draw_fingers(img, anchor, spacing):
    skin = np.array([random.randint(60, 190), random.randint(90, 200), random.randint(130, 245)], float)
    for _ in range(random.randint(1, 4)):
        center = anchor + np.random.normal(0, spacing * 0.8, 2)
        axes = (int(spacing * random.uniform(0.3, 0.6)), int(spacing * random.uniform(0.8, 2.2)))
        color = (skin * random.uniform(0.8, 1.1)).clip(0, 255).tolist()
        cv2.ellipse(img, tuple(int(v) for v in center), axes, random.uniform(0, 180), 0, 360, color, -1, cv2.LINE_AA)


def degrade(img):
    """Camera effects: white balance, exposure, blur, noise, JPEG."""
    img = img.astype(np.float32)
    img *= np.array([random.uniform(0.8, 1.2) for _ in range(3)]) * random.uniform(0.6, 1.3)
    img = img.clip(0, 255).astype(np.uint8)
    if random.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.3, 1.5))
    if random.random() < 0.15:
        k = random.randint(5, 15)
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2] = 1 / k
        rot = cv2.getRotationMatrix2D((k / 2, k / 2), random.uniform(0, 180), 1)
        img = cv2.filter2D(img, -1, cv2.warpAffine(kernel, rot, (k, k)))
    noise = np.random.normal(0, random.uniform(0, 8), img.shape)
    img = (img + noise).clip(0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(35, 95)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- one sample

def render_sample(bg_files):
    """Returns (image, labels); labels are (box xyxy, 9x2 keypoints, 9 visibilities)."""
    w = 640
    h = random.choice([360, 480])
    img = random_background(h, w, bg_files)
    if random.random() < NEGATIVE_FRACTION:
        return degrade(img), []

    cube = Cube()
    for move in random_scramble(random.randint(0, 30)):
        cube.apply(move)

    R = sample_pose()
    f = w * random.uniform(0.8, 1.6)
    size_px = random.uniform(0.2, 0.8) * h
    dist = f * 3 / size_px
    px, py = random.uniform(0.15, 0.85) * w, random.uniform(0.15, 0.85) * h
    t = np.array([(px - w / 2) * dist / f, (py - h / 2) * dist / f, dist])

    def project(p):
        q = R @ p + t
        return np.array([f * q[0] / q[2] + w / 2, f * q[1] / q[2] + h / 2])

    palette = random_palette()
    stickerless = random.random() < 0.2
    half = random.uniform(0.47, 0.49) if stickerless else random.uniform(0.36, 0.46)
    body = np.full(3, random.randint(5, 60)) if random.random() < 0.9 else np.full(3, random.randint(180, 240))
    light = np.array([random.uniform(-1, 1), random.uniform(-1, 1), -1.0])
    light /= np.linalg.norm(light)
    ambient = random.uniform(0.35, 0.75)

    labels, sticker_draws = [], []
    for normal, up in FACE_UP.items():
        n, up = np.array(normal, float), np.array(up, float)
        center = 1.5 * n
        n_cam, c_cam = R @ n, R @ center + t
        if np.dot(n_cam, c_cam) >= 0:
            continue  # facing away
        frontality = -np.dot(n_cam, c_cam / np.linalg.norm(c_cam))
        right = np.cross(up, n)
        shade = ambient + (1 - ambient) * max(0.0, float(-np.dot(n_cam, light)))

        corners = [project(center + 1.5 * (a * up + b * right)) for a, b in ((1, -1), (1, 1), (-1, 1), (-1, -1))]
        fill(img, corners, body * shade)

        colors = cube.face(normal)
        keypoints = []
        for k, name in enumerate(colors):
            r, c = divmod(k, 3)
            s_center = center + up * (1 - r) + right * (c - 1)
            quad = [project(s_center + half * (a * up + b * right)) for a, b in ((1, -1), (1, 1), (-1, 1), (-1, -1))]
            color = palette[name] * shade + np.random.normal(0, 4, 3)
            fill(img, quad, color.clip(0, 255))
            keypoints.append(project(s_center))
            sticker_draws.append(keypoints[-1])

        if frontality >= MIN_FRONTALITY:
            labels.append((np.array(corners), np.array(keypoints)))

    spacing = size_px / 4
    if sticker_draws and random.random() < 0.3:
        for _ in range(random.randint(1, 3)):
            draw_glare(img, random.choice(sticker_draws), spacing)
    if sticker_draws and random.random() < 0.5:
        draw_fingers(img, random.choice(sticker_draws), spacing)

    out = []
    for corners, keypoints in labels:
        keypoints = canonical_order(keypoints)
        inside = (keypoints[:, 0] >= 0) & (keypoints[:, 0] < w) & (keypoints[:, 1] >= 0) & (keypoints[:, 1] < h)
        if inside.sum() < MIN_VISIBLE_KEYPOINTS:
            continue
        x0, y0 = corners.min(axis=0).clip(0, [w, h])
        x1, y1 = corners.max(axis=0).clip(0, [w, h])
        out.append(((x0, y0, x1, y1), keypoints, np.where(inside, 2, 0)))
    return degrade(img), out


def label_line(box, keypoints, visibility, w, h):
    x0, y0, x1, y1 = box
    parts = [0, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
    for (x, y), v in zip(keypoints, visibility):
        parts += [x / w, y / h, int(v)] if v else [0, 0, 0]
    return " ".join(f"{p:.6f}" if isinstance(p, float) else str(p) for p in parts)


# ---------------------------------------------------------------- dataset

def generate(out_dir, n_train, n_val, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    out_dir = Path(out_dir)
    bg_files = sorted(p for p in BACKGROUND_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for split, n in (("train", n_train), ("val", n_val)):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, labels = render_sample(bg_files)
            h, w = img.shape[:2]
            cv2.imwrite(str(out_dir / "images" / split / f"{i:06d}.jpg"), img)
            lines = [label_line(*lab, w, h) for lab in labels]
            (out_dir / "labels" / split / f"{i:06d}.txt").write_text("\n".join(lines))
            if (i + 1) % 500 == 0:
                print(f"{split}: {i + 1}/{n}")
    yaml = out_dir / "data.yaml"
    yaml.write_text(
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "kpt_shape: [9, 3]\n"
        f"flip_idx: {FLIP_IDX}\n"
        "names:\n  0: face\n"
    )
    print(f"wrote {n_train} train / {n_val} val images to {out_dir}")
    return yaml


def preview(n, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bg_files = sorted(p for p in BACKGROUND_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for i in range(n):
        img, labels = render_sample(bg_files)
        for box, keypoints, vis in labels:
            cv2.rectangle(img, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]), (0, 255, 0), 1)
            for k, ((x, y), v) in enumerate(zip(keypoints, vis)):
                if v:
                    cv2.putText(img, str(k), (int(x) - 4, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        cv2.imwrite(str(out_dir / f"preview_{i:02d}.jpg"), img)
    print(f"wrote {n} previews to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "datasets" / "cube_pose"))
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--val", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", type=int, default=0, help="only write N labelled preview images")
    args = parser.parse_args()
    if args.preview:
        preview(args.preview, Path(args.out) / "preview")
    else:
        generate(args.out, args.train, args.val, args.seed)

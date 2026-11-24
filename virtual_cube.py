"""Virtual 3x3 cube you can turn and spin, like a real one.

Controls (click the window first so it has keyboard focus):
  u d l r f b     turn that face clockwise (as seen when looking at that face)
  m e s           turn a middle slice (M follows L, E follows D, S follows F)
  Shift + letter  turn counter-clockwise (or press ' first, then the letter)
                  Caps Lock has no effect on turn direction
  mouse drag      rotate the whole cube freely (left/right, up/down)

Turns are relative to how you are holding the cube, like on a real one: "u" turns
whichever face is currently on top of the screen, "f" the one facing you, etc.
  space           scramble
  backspace       undo last turn
  0               reset to solved
  q / Esc         quit
"""
import ctypes
import itertools
import math
import random
import sys
import time
from collections import deque

import cv2
import numpy as np

from display import DISPLAY_BGR

# Outward face normal -> sticker color (standard scheme: white top, green front).
FACE_COLORS = {
    (0, 1, 0): "white", (0, -1, 0): "yellow",
    (0, 0, 1): "green", (0, 0, -1): "blue",
    (1, 0, 0): "red", (-1, 0, 0): "orange",
}
# Move letter -> (axis 0=x/1=y/2=z, layer coordinate, quarter turns about +axis for a clockwise turn).
# x points right, y up, z toward the viewer (front face).
MOVES = {
    "U": (1, 1, -1), "D": (1, -1, 1), "E": (1, 0, 1),
    "R": (0, 1, -1), "L": (0, -1, 1), "M": (0, 0, 1),
    "F": (2, 1, -1), "B": (2, -1, 1), "S": (2, 0, -1),
}
# For reading a face as a 3x3 grid: normal -> "up" direction when looking at that face.
FACE_UP = {
    (0, 0, 1): (0, 1, 0), (0, 0, -1): (0, 1, 0), (1, 0, 0): (0, 1, 0), (-1, 0, 0): (0, 1, 0),
    (0, 1, 0): (0, 0, -1), (0, -1, 0): (0, 0, 1),
}

TURN_TIME = 0.18          # seconds per animated quarter turn
SCRAMBLE_TURN_TIME = 0.07
SCRAMBLE_LENGTH = 25
WINDOW = "Virtual Cube"
SIZE = 720
CAM_DIST = 9.0
FOCAL = 1.2 * SIZE
BODY_BGR = (25, 25, 25)
BG_BGR = (60, 50, 45)
STICKER_HALF = 0.42       # sticker half-size (cubie half-size is 0.5)
LIGHT = np.array([-0.3, 0.5, 1.0]) / np.linalg.norm([-0.3, 0.5, 1.0])


def axis_rotation(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def parse_move(move):
    """'R' -> (axis, layer, quarter turns); "R'" turns the other way."""
    axis, layer, quarters = MOVES[move[0]]
    return axis, layer, -quarters if move.endswith("'") else quarters


def inverse(move):
    return move[:-1] if move.endswith("'") else move + "'"


def snap(v, exclude=()):
    """Nearest world axis direction to vector v, e.g. (0, -1, 0)."""
    axis = max((a for a in range(3) if a not in exclude), key=lambda a: abs(v[a]))
    n = np.zeros(3, int)
    n[axis] = 1 if v[axis] > 0 else -1
    return n


def view_turn(letter, prime, view):
    """Turn for a move letter relative to the current view -> world (axis, layer, quarters)."""
    up = snap(view.T @ (0, 1, 0))  # world direction currently pointing up on screen
    front = snap(view.T @ (0, 0, 1), exclude=[int(np.argmax(np.abs(up)))])
    right = np.cross(up, front)
    face = {"M": "L", "E": "D", "S": "F"}.get(letter, letter)
    n = {"U": up, "D": -up, "F": front, "B": -front, "R": right, "L": -right}[face]
    axis = int(np.argmax(np.abs(n)))
    sign = int(n[axis])
    layer = 0 if letter in "MES" else sign
    quarters = -sign  # clockwise seen from outside = -90 degrees about the outward normal
    return axis, layer, -quarters if prime else quarters


def orthonormalize(m):
    u, _, vt = np.linalg.svd(m)
    return u @ vt


class Cubie:
    def __init__(self, pos):
        self.pos = np.array(pos)
        # Sticker normals are stored in world space and rotate with the cubie.
        self.stickers = {n: color for n, color in FACE_COLORS.items() if np.dot(n, pos) == 1}


class Cube:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cubies = [Cubie(p) for p in itertools.product((-1, 0, 1), repeat=3)]

    def apply(self, move):
        """Apply a move in world notation, e.g. "R" or "U'"."""
        self.apply_turn(*parse_move(move))

    def apply_turn(self, axis, layer, quarters):
        rot = np.rint(axis_rotation(axis, quarters * math.pi / 2)).astype(int)
        for c in self.cubies:
            if c.pos[axis] == layer:
                c.pos = rot @ c.pos
                c.stickers = {tuple(int(x) for x in rot @ n): col for n, col in c.stickers.items()}

    def is_solved(self):
        faces = {}
        for c in self.cubies:
            for n, color in c.stickers.items():
                faces.setdefault(n, set()).add(color)
        return all(len(colors) == 1 for colors in faces.values())

    def center_color(self, normal):
        """Color of the center sticker on the face pointing in world direction `normal`."""
        normal = tuple(int(x) for x in normal)
        return next(c.stickers[normal] for c in self.cubies if tuple(c.pos) == normal)

    def face(self, normal):
        """9 color names of one face, row-major, as seen looking straight at it
        (same layout the camera detector reports)."""
        n, up = np.array(normal), np.array(FACE_UP[tuple(normal)])
        right = np.cross(up, n)
        by_pos = {tuple(c.pos): c for c in self.cubies}
        return [by_pos[tuple(n + up * (1 - r) + right * (col - 1))].stickers[tuple(normal)]
                for r in range(3) for col in range(3)]


def random_scramble(length=SCRAMBLE_LENGTH):
    moves, last = [], None
    for _ in range(length):
        face = random.choice([f for f in "UDLRFB" if f != last])
        moves.append(face + random.choice(["", "'"]))
        last = face
    return moves


# ---------------------------------------------------------------- rendering

def project(p):
    d = CAM_DIST - p[2]
    return SIZE / 2 + FOCAL * p[0] / d, SIZE / 2 - FOCAL * p[1] / d


def fill(img, pts, color):
    pts = np.round(np.array(pts) * 16).astype(np.int32)  # 4 bits of sub-pixel precision
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA, shift=4)


def draw_cubie(img, cubie, transform):
    cam = np.array([0.0, 0.0, CAM_DIST])
    for axis, sign in itertools.product(range(3), (-1, 1)):
        n = np.zeros(3)
        n[axis] = sign
        u, v = np.eye(3)[(axis + 1) % 3], np.eye(3)[(axis + 2) % 3]
        center = cubie.pos + 0.5 * n
        n_view = transform @ n
        if np.dot(n_view, cam - transform @ center) <= 0:
            continue  # facing away from the camera
        quad = lambda c, h: [project(transform @ (c + h * (a * u + b * v)))
                             for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        fill(img, quad(center, 0.5), BODY_BGR)
        color = cubie.stickers.get(tuple(int(x) for x in n))
        if color:
            shade = 0.6 + 0.4 * max(0.0, float(np.dot(n_view, LIGHT)))
            fill(img, quad(center + 0.002 * n, STICKER_HALF),
                 tuple(int(ch * shade) for ch in DISPLAY_BGR[color]))


def render(cube, view, turning=None):
    """turning: (axis, layer, angle) of a layer mid-animation, or None."""
    img = np.full((SIZE, SIZE, 3), BG_BGR, np.uint8)
    cam_world = view.T @ np.array([0.0, 0.0, CAM_DIST])
    axis, layer, angle = turning if turning else (1, None, 0.0)
    # Painter's algorithm: the three slabs along the turning axis are separated by planes,
    # so draw the slabs farthest from the camera first; same idea for cubies inside a slab.
    for k in sorted((-1, 0, 1), key=lambda k: -abs(k - cam_world[axis])):
        slab_rot = axis_rotation(axis, angle) if k == layer else np.eye(3)
        cam_slab = slab_rot.T @ cam_world
        others = [a for a in range(3) if a != axis]
        group = [c for c in cube.cubies if c.pos[axis] == k]
        group.sort(key=lambda c: -sum(abs(c.pos[a] - cam_slab[a]) for a in others))
        for c in group:
            draw_cubie(img, c, view @ slab_rot)
    return img


def caps_lock_on():
    """Whether Caps Lock is on (Windows only; elsewhere assume off)."""
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.GetKeyState(0x14) & 1)


def put_text(img, text, org, color=(230, 230, 230), scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------- app

def main():
    print(__doc__)
    cube = Cube()
    # World -> view rotation. Starts showing the top, front and right faces.
    view = axis_rotation(0, math.radians(25)) @ axis_rotation(1, math.radians(-35))
    queue, history = deque(), []  # queue: (label, turn, duration, record); history: (label, turn)
    anim = None  # (label, turn, start time, duration, record in history)
    prime_next = False
    drag = {"last": None}

    def on_mouse(event, x, y, flags, _):
        nonlocal view
        if event == cv2.EVENT_LBUTTONDOWN:
            drag["last"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drag["last"] = None
        elif event == cv2.EVENT_MOUSEMOVE and drag["last"] and flags & cv2.EVENT_FLAG_LBUTTON:
            dx, dy = x - drag["last"][0], y - drag["last"][1]
            # Rotate about the screen's vertical and horizontal axes - no limits, any orientation.
            view = axis_rotation(1, dx * 0.01) @ axis_rotation(0, dy * 0.01) @ view
            view = orthonormalize(view)  # keep rounding errors from distorting the cube
            drag["last"] = (x, y)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        now = time.perf_counter()
        if anim is None and queue:
            label, turn, duration, record = queue.popleft()
            anim = (label, turn, now, duration, record)

        turning = None
        if anim:
            label, turn, start, duration, record = anim
            t = (now - start) / duration
            if t >= 1:
                cube.apply_turn(*turn)
                if record:
                    history.append((label, turn))
                anim = None
            else:
                axis, layer, quarters = turn
                eased = t * t * (3 - 2 * t)
                turning = (axis, layer, quarters * math.pi / 2 * eased)

        img = render(cube, view, turning)

        busy = anim is not None or bool(queue)
        if cube.is_solved() and not busy:
            put_text(img, "SOLVED", (12, 30), (120, 255, 120), 0.8)
        else:
            put_text(img, f"moves: {len(history)}", (12, 30), scale=0.7)
        put_text(img, " ".join(label for label, _ in history[-14:]), (12, 58), (190, 190, 190))
        top = snap(view.T @ (0, 1, 0))
        front = snap(view.T @ (0, 0, 1), exclude=[int(np.argmax(np.abs(top)))])
        put_text(img, f"top: {cube.center_color(top)}   front: {cube.center_color(front)}",
                 (SIZE - 260, 30), (190, 190, 190))
        if prime_next:
            put_text(img, "next turn: counter-clockwise", (12, 84), (120, 200, 255))
        put_text(img, "udlrfb mes: turn  Shift: reverse  drag: rotate cube", (12, SIZE - 36))
        put_text(img, "space: scramble  backspace: undo  0: reset  q: quit", (12, SIZE - 14))
        cv2.imshow(WINDOW, img)

        key = cv2.waitKey(10)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break  # window closed with the X button
        if key == -1:
            continue
        key &= 0xFF
        ch = chr(key)
        if key == 27 or ch.lower() == "q":
            break
        if ch.upper() in MOVES:
            # Shift makes the letter uppercase, but Caps Lock flips the case too,
            # so undo the Caps Lock flip: only Shift decides the direction.
            shifted = ch.isupper() != caps_lock_on()
            prime = shifted != prime_next
            label = ch.upper() + ("'" if prime else "")
            queue.append((label, view_turn(ch.upper(), prime, view), TURN_TIME, True))
            prime_next = False
        elif ch == "'":
            prime_next = not prime_next
        elif ch == " " and not busy:
            queue.extend((m, parse_move(m), SCRAMBLE_TURN_TIME, True) for m in random_scramble())
        elif key == 8 and history and not busy:  # backspace
            label, (axis, layer, quarters) = history.pop()
            queue.append((inverse(label), (axis, layer, -quarters), TURN_TIME, False))
        elif ch == "0":
            cube.reset()
            queue.clear()
            history.clear()
            anim = None

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

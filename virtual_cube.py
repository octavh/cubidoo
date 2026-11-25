"""Virtual 3x3 cube you can turn and spin, like a real one.

Controls (click the window first so it has keyboard focus):
  u d l r f b     turn that face clockwise (as seen when looking at that face)
  m e s           turn a middle slice (M follows L, E follows D, S follows F)
  Shift + letter  turn counter-clockwise (or press ' first, then the letter)
                  Caps Lock has no effect on turn direction
  mouse drag      rotate the whole cube freely (left/right, up/down)

Turns are relative to how you are holding the cube, like on a real one: "u" turns
whichever face is currently on top of the screen, "f" the one facing you, etc.
  space           random scramble
  Enter           type your own scramble (e.g. R U R' U2 F), Enter again to apply, Esc to cancel
  Ctrl+V or p     apply the scramble you copied (while typing, Ctrl+V pastes into the box)
  backspace       undo last turn
  0               reset to solved
  q / Esc         quit

Start from your own scramble:   python virtual_cube.py "R U R' U2 F"
Scrambles use standard notation with white on top and green in front.
"""
import ctypes
import itertools
import math
import random
import re
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

TURN_TIME = 0.14          # seconds per animated quarter turn
SCRAMBLE_TURN_TIME = 0.06
MIN_TURN_TIME = 0.04      # fastest a turn plays when several are waiting in line
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


def parse_scramble(text):
    """ "R U2 F' M" -> ["R", "U", "U", "F'", "M"]. Raises ValueError on anything unknown."""
    text = text.replace("’", "'").replace("′", "'").replace("`", "'")  # pasted primes
    moves = []
    for token in re.findall(r"[A-Za-z](?:2'|'2|2|')?|\S", text):
        face = token[0].upper()
        if face not in MOVES:
            raise ValueError(f"unknown move '{token}' (use U D L R F B M E S with ' or 2)")
        if "2" in token:
            moves += [face, face]
        else:
            moves.append(face + ("'" if "'" in token else ""))
    return moves


def random_scramble(length=SCRAMBLE_LENGTH):
    moves, last = [], None
    for _ in range(length):
        face = random.choice([f for f in "UDLRFB" if f != last])
        moves.append(face + random.choice(["", "'"]))
        last = face
    return moves


# ---------------------------------------------------------------- rendering

# The 6 face directions of a cubie: outward normal and two in-face axes.
DIRS = [(a, s) for a in range(3) for s in (-1, 1)]
DIR_N = np.array([np.eye(3)[a] * s for a, s in DIRS])
DIR_U = np.array([np.eye(3)[(a + 1) % 3] for a, _ in DIRS])
DIR_V = np.array([np.eye(3)[(a + 2) % 3] for a, _ in DIRS])
SQUARE = np.array([(1, 1), (1, -1), (-1, -1), (-1, 1)], float)
SHADED = {name: np.array(bgr, float) for name, bgr in DISPLAY_BGR.items()}
BACKGROUND = np.full((SIZE, SIZE, 3), BG_BGR, np.uint8)


def render(cube, view, turning=None):
    """turning: (axis, layer, angle) of a layer mid-animation, or None.

    All geometry is computed with a few vectorized numpy calls; only the polygon
    filling loops in Python."""
    img = BACKGROUND.copy()  # much faster than np.full with a color every frame
    cam_world = view.T @ np.array([0.0, 0.0, CAM_DIST])
    axis, layer, angle = turning if turning else (1, None, 0.0)
    turn_rot = axis_rotation(axis, angle)

    # Painter's algorithm: the three slabs along the turning axis are separated by planes,
    # so draw the slabs farthest from the camera first; same idea for cubies inside a slab.
    pos, dirs, moving, colors = [], [], [], []
    for k in sorted((-1, 0, 1), key=lambda k: -abs(k - cam_world[axis])):
        rot = turn_rot if k == layer else np.eye(3)
        cam_slab = rot.T @ cam_world
        others = [a for a in range(3) if a != axis]
        group = [c for c in cube.cubies if c.pos[axis] == k]
        group.sort(key=lambda c: -sum(abs(c.pos[a] - cam_slab[a]) for a in others))
        for c in group:
            for d, (a, s) in enumerate(DIRS):
                # Only outer faces can be seen, plus the inner faces a turning layer uncovers.
                if c.pos[a] != s and not (turning and a == axis):
                    continue
                pos.append(c.pos)
                dirs.append(d)
                moving.append(k == layer)
                n = [0, 0, 0]
                n[a] = s
                colors.append(c.stickers.get(tuple(n)))

    pos, dirs = np.array(pos, float), np.array(dirs)
    mats = np.where(np.array(moving)[:, None, None], view @ turn_rot, view)  # (F, 3, 3)
    n, u, v = DIR_N[dirs], DIR_U[dirs], DIR_V[dirs]
    centers = pos + 0.5 * n

    def corners(center, half):  # (F, 4, 3) in view space
        pts = center[:, None] + half * (SQUARE[None, :, :1] * u[:, None] + SQUARE[None, :, 1:] * v[:, None])
        return np.einsum("fij,fkj->fki", mats, pts)

    def project(p):  # (F, 4, 3) -> (F, 4, 2) pixel coords, x16 for sub-pixel drawing
        d = CAM_DIST - p[..., 2]
        xy = np.stack([SIZE / 2 + FOCAL * p[..., 0] / d, SIZE / 2 - FOCAL * p[..., 1] / d], -1)
        return np.round(xy * 16).astype(np.int32)

    n_view = np.einsum("fij,fj->fi", mats, n)
    c_view = np.einsum("fij,fj->fi", mats, centers)
    visible = np.einsum("fi,fi->f", n_view, np.array([0, 0, CAM_DIST]) - c_view) > 0
    shade = 0.6 + 0.4 * np.clip(n_view @ LIGHT, 0, None)
    body = project(corners(centers, 0.5))
    sticker = project(corners(centers + 0.002 * n, STICKER_HALF))

    for f in np.flatnonzero(visible):
        cv2.fillPoly(img, [body[f]], BODY_BGR, cv2.LINE_AA, shift=4)
        if colors[f]:
            color = tuple(int(x) for x in SHADED[colors[f]] * shade[f])
            cv2.fillPoly(img, [sticker[f]], color, cv2.LINE_AA, shift=4)
    return img


def read_clipboard():
    """Text on the clipboard, or "" if there is none."""
    if sys.platform == "win32":
        k32, u32 = ctypes.windll.kernel32, ctypes.windll.user32
        u32.GetClipboardData.restype = ctypes.c_void_p
        k32.GlobalLock.argtypes = [ctypes.c_void_p]
        k32.GlobalLock.restype = ctypes.c_void_p
        k32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        if not u32.OpenClipboard(None):
            return ""
        try:
            handle = u32.GetClipboardData(13)  # CF_UNICODETEXT
            if not handle:
                return ""
            ptr = k32.GlobalLock(handle)
            try:
                return ctypes.wstring_at(ptr) if ptr else ""
            finally:
                k32.GlobalUnlock(handle)
        finally:
            u32.CloseClipboard()
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        try:
            return root.clipboard_get()
        finally:
            root.destroy()
    except Exception:
        return ""


def clean_pasted(text):
    """One line of plain text: curly primes -> ', newlines/tabs -> single spaces."""
    text = text.replace("’", "'").replace("′", "'").replace("`", "'")
    return " ".join(text.split())


def ctrl_held():
    return sys.platform == "win32" and bool(ctypes.windll.user32.GetKeyState(0x11) & 0x8000)


def caps_lock_on():
    """Whether Caps Lock is on (Windows only; elsewhere assume off)."""
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.GetKeyState(0x14) & 1)


def put_text(img, text, org, color=(230, 230, 230), scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------- app

def main(scramble=""):
    print(__doc__)
    cube = Cube()
    # World -> view rotation. Starts showing the top, front and right faces.
    view = axis_rotation(0, math.radians(25)) @ axis_rotation(1, math.radians(-35))
    queue, history = deque(), []  # queue: (label, turn, duration, record); history: (label, turn)
    anim = None  # (label, turn, start time, duration, record in history)
    idle_since = time.perf_counter()  # when the last turn ended (or now, if idle)
    prime_next = False
    drag = {"last": None}
    typing = None          # scramble text being typed, or None when not typing
    message = None         # (error text, show until time)
    scramble_label = ""    # the scramble the cube was started from, shown on screen

    def start_from(text):
        """Reset, then apply a scramble instantly. Raises ValueError if it can't be read."""
        nonlocal anim, scramble_label
        moves = parse_scramble(text)  # parse first, so a typo leaves the cube untouched
        cube.reset()
        for m in moves:
            cube.apply(m)
        queue.clear()
        history.clear()
        anim = None
        scramble_label = " ".join(text.split())

    if scramble.strip():
        start_from(scramble)

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
        turning = None
        # Finish any turns whose time is up and start the next one in the same frame,
        # timed from when the previous one ended, so queued turns flow without pauses.
        while anim or queue:
            if anim is None:
                label, turn, duration, record = queue.popleft()
                # Turns waiting in line play faster, so the cube keeps up with fast typing.
                duration = max(MIN_TURN_TIME, duration / (1 + len(queue)))
                anim = (label, turn, max(now - duration, idle_since), duration, record)
            label, turn, start, duration, record = anim
            t = (now - start) / duration
            if t < 1:
                axis, layer, quarters = turn
                eased = 1 - (1 - t) ** 3  # ease-out: moves right away, settles gently
                turning = (axis, layer, quarters * math.pi / 2 * eased)
                break
            cube.apply_turn(*turn)
            if record:
                history.append((label, turn))
            anim = None
            idle_since = start + duration  # next queued turn starts exactly when this ended
        if anim is None and not queue:
            idle_since = now

        img = render(cube, view, turning)

        if typing is not None:
            cv2.rectangle(img, (0, SIZE - 130), (SIZE, SIZE - 60), (90, 60, 30), -1)
            cursor = "_" if int(now * 2) % 2 == 0 else " "
            put_text(img, "TYPING A SCRAMBLE - keys won't turn the cube", (12, SIZE - 108),
                     (120, 220, 255), 0.55)
            shown = typing if len(typing) <= 55 else "..." + typing[-52:]  # long pastes: show the end
            put_text(img, f"Scramble: {shown}{cursor}", (12, SIZE - 84), scale=0.6)
            put_text(img, "Enter: apply   Ctrl+V: paste   Esc: cancel   (white top, green front)",
                     (12, SIZE - 66), (200, 200, 200), 0.45)
        elif message and now < message[1]:
            put_text(img, message[0], (12, SIZE - 70), (100, 100, 255), 0.55)

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
        if scramble_label:
            put_text(img, f"scramble: {scramble_label}", (12, 84), (170, 170, 170), 0.45)
        if prime_next:
            put_text(img, "next turn: counter-clockwise", (12, 110), (120, 200, 255))
        put_text(img, "udlrfb mes: turn  Shift: reverse  drag: rotate cube  Ctrl+V / p: paste scramble",
                 (12, SIZE - 36), scale=0.45)
        put_text(img, "space: random scramble  Enter: type scramble  backspace: undo  0: reset  q: quit",
                 (12, SIZE - 14), scale=0.45)
        cv2.imshow(WINDOW, img)

        key = cv2.waitKey(1)  # short wait = more frames per second = smoother motion
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break  # window closed with the X button
        if key == -1:
            continue
        key &= 0xFF
        ch = chr(key)
        # Ctrl+V arrives as code 22, or as "v" with Ctrl held, depending on the system.
        paste = key == 22 or (ch.lower() == "v" and ctrl_held())

        if typing is not None:  # typing a scramble: keys type text instead of turning
            if paste:
                typing += clean_pasted(read_clipboard())
            elif key in (13, 10):  # Enter
                try:
                    start_from(typing)
                except ValueError as e:
                    message = (str(e), now + 4)
                typing = None
            elif key == 27:  # Esc
                typing = None
            elif key == 8:  # Backspace
                typing = typing[:-1]
            elif 32 <= key < 127:
                typing += ch
            continue

        if key == 27 or ch.lower() == "q":
            break
        if key in (13, 10):  # Enter: start typing a scramble
            typing = ""
        elif paste or ch.lower() == "p":  # apply the copied scramble right away
            pasted = clean_pasted(read_clipboard())
            if not pasted:
                message = ("Clipboard is empty - copy a scramble first", now + 4)
            else:
                try:
                    start_from(pasted)
                except ValueError as e:
                    message = (f"Pasted text isn't a scramble: {e}", now + 4)
        elif ch.upper() in MOVES:
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
            scramble_label = ""

    cv2.destroyAllWindows()


if __name__ == "__main__":
    scramble = " ".join(sys.argv[1:])
    try:
        parse_scramble(scramble)
    except ValueError as e:
        raise SystemExit(f"Invalid scramble: {e}")
    main(scramble)

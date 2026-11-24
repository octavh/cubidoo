import sys
from collections import Counter, deque

import cv2

from color_detector import ColorDetector
from display import draw_detection, render_grid
from face_detector import detect_face

CALIBRATION_KEYS = {
    ord("w"): "white",
    ord("y"): "yellow",
    ord("r"): "red",
    ord("o"): "orange",
    ord("g"): "green",
    ord("b"): "blue",
}

HISTORY_FRAMES = 7     # recent detections that vote on each sticker's color
LOST_GRACE_FRAMES = 8  # keep showing "detected" this many frames after losing the face
FORGET_FRAMES = 15     # clear vote history after this many frames without a face


def open_camera():
    # DirectShow opens much faster than the default backend on Windows.
    if sys.platform == "win32":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(0)


def vote(history):
    """Most common color per sticker position across recent detections."""
    return [Counter(colors[k] for colors in history).most_common(1)[0][0] for k in range(9)]


def main():
    cap = open_camera()
    if not cap.isOpened():
        raise SystemExit("Could not open camera. On macOS, grant camera access to your terminal/IDE "
                         "in System Settings > Privacy & Security > Camera.")

    color_detector = ColorDetector()
    last_colors = [None] * 9
    history = deque(maxlen=HISTORY_FRAMES)
    frames_since_face = FORGET_FRAMES
    mirror = True

    print(__doc__)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        # Detect on the raw (unmirrored) frame so the grid matches the real face layout.
        face, candidates = detect_face(frame, color_detector)
        if face is not None:
            if frames_since_face >= FORGET_FRAMES:
                history.clear()  # face was gone a while - probably a different face now
            history.append(face.colors)
            last_colors = vote(history)
            frames_since_face = 0
        else:
            frames_since_face += 1
        detected = frames_since_face <= LOST_GRACE_FRAMES

        preview = frame.copy()
        draw_detection(preview, face, candidates)
        if mirror:
            preview = cv2.flip(preview, 1)
        cv2.putText(preview, "q: quit  m: mirror  w/y/r/o/g/b: calibrate center  x: reset",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Camera", preview)
        cv2.imshow("Cube Face", render_grid(last_colors, detected))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            mirror = not mirror
        elif key == ord("x"):
            color_detector.reset()
            history.clear()
            print("calibration reset")
        elif key in CALIBRATION_KEYS:
            if face is None:
                print("No face detected - show the cube to calibrate.")
            else:
                color_detector.calibrate(CALIBRATION_KEYS[key], face.patches[4])
                history.clear()  # old votes used the previous calibration

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""cubidoo - read the colors of one Rubik's cube face from your webcam.

Show a face of the cube to the camera anywhere in the frame. Whenever a face
is detected, the "Cube Face" window is updated with the colors it reads.
When no face is visible, the grid keeps the last reading.

Keys:
  q / ESC   quit
  m         toggle mirrored preview
  w y r o g b   calibrate that color using the CENTER sticker of the detected face
  x         reset calibration to defaults
"""

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


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera. On macOS, grant camera access to your terminal/IDE "
                         "in System Settings > Privacy & Security > Camera.")

    color_detector = ColorDetector()
    last_colors = [None] * 9
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
            last_colors = face.colors

        preview = frame.copy()
        draw_detection(preview, face, candidates)
        if mirror:
            preview = cv2.flip(preview, 1)
        cv2.putText(preview, "q: quit  m: mirror  w/y/r/o/g/b: calibrate center  x: reset",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Camera", preview)
        cv2.imshow("Cube Face", render_grid(last_colors, face is not None))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            mirror = not mirror
        elif key == ord("x"):
            color_detector.reset()
            print("calibration reset")
        elif key in CALIBRATION_KEYS:
            if face is None:
                print("No face detected - show the cube to calibrate.")
            else:
                color_detector.calibrate(CALIBRATION_KEYS[key], face.patches[4])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

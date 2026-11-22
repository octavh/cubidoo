# Cubidoo
Interactive Cube detector made with Machine Learning

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r libraries.txt
.venv/bin/python main.py
```

Show one face of the cube to the camera, anywhere in the frame. When a face is detected (green outline),
the **Cube Face** window shows its 3x3 colors (as seen looking straight at the face). When no face is
visible, the grid keeps the last reading.

Keys: `q` quit · `m` toggle mirror · `w y r o g b` calibrate that color from the center sticker · `x` reset calibration

## Files

- `main.py` – camera loop and key handling
- `face_detector.py` – finds the 3x3 sticker grid anywhere in the frame
- `color_detector.py` – classifies each sticker's color (with calibration)
- `display.py` – draws the camera overlay and the 3x3 grid window

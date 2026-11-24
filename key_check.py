"""Shows what the keyboard sends to an OpenCV window. Changes nothing, sends no keys.

Run it, click the black window, press: r   then   Shift+r   then   q (or wait 20 s).
"""
import ctypes
import time

import cv2
import numpy as np

user32 = ctypes.windll.user32
user32.GetKeyState.restype = ctypes.c_short
user32.GetAsyncKeyState.restype = ctypes.c_short

WIN = "key check - click me, press r then Shift+r"
cv2.namedWindow(WIN)
cv2.imshow(WIN, np.zeros((150, 520, 3), np.uint8))
print("Caps Lock is", "ON" if user32.GetKeyState(0x14) & 1 else "OFF")
print("Click the black window, then press r, then Shift+r, then q.\n")

r_was_down = shift_was_down = False
end = time.time() + 20
while time.time() < end:
    key = cv2.waitKey(10)
    if key != -1:
        shift = bool(user32.GetKeyState(0x10) & 0x8000)
        print(f"OpenCV got key {key} ({chr(key & 0xFF)!r}), Shift held: {shift}")
        if chr(key & 0xFF).lower() == "q":
            break
    shift_down = bool(user32.GetAsyncKeyState(0x10) & 0x8000)
    r_down = bool(user32.GetAsyncKeyState(ord("R")) & 0x8000)
    if shift_down != shift_was_down:
        print("   keyboard: Shift", "pressed" if shift_down else "released")
    if r_down and not r_was_down:
        print(f"   keyboard: R pressed (Shift held: {shift_down})")
    r_was_down, shift_was_down = r_down, shift_down
cv2.destroyAllWindows()

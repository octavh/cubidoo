"""Classify the color of a cube sticker from an HSV image patch."""

import numpy as np

MIN_VALUE = 60             # darker than this -> not a sticker
STICKER_MIN_SAT = 70       # colored stickers must be at least this saturated
WHITE_MAX_SAT = 60         # white: low saturation...
WHITE_MIN_VALUE = 120      # ...and bright

# Default reference hues (OpenCV scale 0-179). Recalibrate at runtime.
DEFAULT_HUES = {
    "red": 177,
    "orange": 10,
    "yellow": 28,
    "green": 65,
    "blue": 108,
}

COLOR_NAMES = ["white", *DEFAULT_HUES]


def hue_distance(a, b):
    d = abs(float(a) - float(b)) % 180
    return min(d, 180 - d)


def circular_hue_mean(hues):
    angles = hues.astype(np.float32) * (2 * np.pi / 180)
    mean = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return (mean * 180 / (2 * np.pi)) % 180


class ColorDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.hues = dict(DEFAULT_HUES)
        self.white_max_sat = WHITE_MAX_SAT
        self.white_min_value = WHITE_MIN_VALUE

    @staticmethod
    def patch_features(hsv_patch):
        hue = circular_hue_mean(hsv_patch[..., 0].ravel())
        sat = float(np.median(hsv_patch[..., 1]))
        val = float(np.median(hsv_patch[..., 2]))
        return hue, sat, val

    def is_sticker(self, hsv_patch):
        """Loose check that a patch looks like a cube sticker (not black/gray stuff)."""
        if hsv_patch.size == 0:
            return False
        _, sat, val = self.patch_features(hsv_patch)
        if val < MIN_VALUE:
            return False
        return sat >= STICKER_MIN_SAT or (sat < self.white_max_sat and val >= self.white_min_value)

    def classify(self, hsv_patch):
        """Best guess, always returns a color name."""
        hue, sat, _ = self.patch_features(hsv_patch)
        if sat < self.white_max_sat:
            return "white"
        return min(self.hues, key=lambda name: hue_distance(hue, self.hues[name]))

    def calibrate(self, name, hsv_patch):
        hue, sat, val = self.patch_features(hsv_patch)
        if name == "white":
            self.white_max_sat = max(WHITE_MAX_SAT, sat + 25)
            self.white_min_value = max(MIN_VALUE, val - 60)
            print(f"calibrated white: sat<{self.white_max_sat:.0f}, val>={self.white_min_value:.0f}")
        else:
            self.hues[name] = hue
            print(f"calibrated {name}: hue={hue:.1f} (sat={sat:.0f}, val={val:.0f})")

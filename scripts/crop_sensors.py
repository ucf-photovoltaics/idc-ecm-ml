"""Level a whole-board scan and cut out the four sensors (U1-U4).

What this does, in plain words
------------------------------
1. Find the board outline (everything that is not the white scanner background).
2. Measure how much the board is rotated, where its centre is, and how big it is.
3. Rotate the board level, scale it to the usual size, and slide it so its centre
   sits in the middle of the scan (a few scans have the board 2-5% smaller).
4. Cut each sensor out using saved box positions (fractions of the picture),
   plus a small extra margin so the ends of the electrodes are not clipped.

Only the small crops are ever built at full resolution, so it stays fast.
"""
import os
import sys

import cv2
import numpy as np

# Box for each sensor as fractions of the (levelled) picture: (left, right, top, bottom).
# One set per board type (the middle number of the board ID). Copied from
# IDC_EM_analysis/Analysis/generators.py.
SENSOR_BOXES = {
    1:  {"U1": (0.1012, 0.3067, 0.576, 0.6601),   "U2": (0.1377, 0.2395, 0.09905, 0.2684),
         "U3": (0.7647, 0.8325, 0.07259, 0.3275), "U4": (0.7736, 0.8254, 0.434, 0.7743)},
    4:  {"U1": (0.09917, 0.3052, 0.5761, 0.6606), "U2": (0.1369, 0.2389, 0.09948, 0.2688),
         "U3": (0.7653, 0.8328, 0.07529, 0.3299), "U4": (0.773, 0.8243, 0.4364, 0.7764)},
    7:  {"U1": (0.1011, 0.3066, 0.5776, 0.6608),  "U2": (0.137, 0.2387, 0.1002, 0.2691),
         "U3": (0.7646, 0.8324, 0.07282, 0.3275), "U4": (0.7737, 0.8255, 0.4342, 0.7739)},
    10: {"U1": (0.09913, 0.3055, 0.5753, 0.6594), "U2": (0.1367, 0.2384, 0.0993, 0.2683),
         "U3": (0.7644, 0.8324, 0.07448, 0.3286), "U4": (0.7727, 0.8243, 0.4351, 0.7745)},
}
MARGIN = 0.08          # extra room around each box, each side: 8% of the box's LONGER side
                       # (so thin sensors like U4 still get a useful amount of room)
# Extra room on ONE side only, where checking real scans showed the buildup runs close to the edge.
# Same units as MARGIN (fraction of the box's longer side). Sides: left, right, top, bottom.
EXTRA_MARGIN = {"U2": {"right": 0.06}, "U3": {"left": 0.06}, "U4": {"left": 0.06}}
# How much of the scan the board usually fills (median over all 98 exposed + 4 pristine scans):
# long side / scan width, short side / scan height.
TYPICAL_BOARD_FRACTION = (0.9909, 0.9964)
REDUCE = 8             # outline is found on a picture 8x smaller (fast, plenty accurate)


def board_outline(small_bgr):
    """Return (centre_x, centre_y, tilt_degrees, size_ratio) of the board in a small picture.

    size_ratio = how much bigger the board must be made to fill the scan as much as usual
    (1.0 = already usual size, 1.05 = board is 5% too small).
    """
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    # Board = coloured or dark pixels. Scanner background = bright and colourless.
    mask = ((hsv[..., 1] > 50) | (hsv[..., 2] < 170)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    board = max(contours, key=cv2.contourArea)
    (cx, cy), (rect_w, rect_h), angle = cv2.minAreaRect(board)
    tilt = (angle + 45) % 90 - 45          # how far from perfectly square, -45..45 degrees
    long_side, short_side = max(rect_w, rect_h), min(rect_w, rect_h)
    small_h, small_w = small_bgr.shape[:2]
    size_ratio = 0.5 * (TYPICAL_BOARD_FRACTION[0] * small_w / long_side
                        + TYPICAL_BOARD_FRACTION[1] * small_h / short_side)
    return cx, cy, tilt, size_ratio


def level_matrix(small_bgr, full_w, full_h):
    """2x3 matrix (full-size pixel units) that levels, resizes and centres the board."""
    cx, cy, tilt, size_ratio = board_outline(small_bgr)
    cx, cy = cx * REDUCE, cy * REDUCE
    m = cv2.getRotationMatrix2D((cx, cy), tilt, size_ratio)
    m[:, 2] += (full_w / 2 - cx, full_h / 2 - cy)
    return m, tilt


def crop_sensors(path, board_type):
    """Return {sensor: BGR crop} and the leveling tilt (degrees) for one scan."""
    small = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)
    full = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = full.shape[:2]
    m, tilt = level_matrix(small, w, h)
    crops = {}
    for sensor, (x1, x2, y1, y2) in SENSOR_BOXES[board_type].items():
        longer = max((x2 - x1) * w, (y2 - y1) * h)       # longer side of the box, in pixels
        mx, my = MARGIN * longer / w, MARGIN * longer / h  # same pixel margin on all four sides
        extra = EXTRA_MARGIN.get(sensor, {})
        ex = lambda side: extra.get(side, 0.0) * longer     # extra pixels on that side
        left, right = round((x1 - mx) * w - ex("left")), round((x2 + mx) * w + ex("right"))
        top, bottom = round((y1 - my) * h - ex("top")), round((y2 + my) * h + ex("bottom"))
        shift = m.copy()
        shift[:, 2] -= (left, top)         # move so the crop's corner becomes (0, 0)
        crops[sensor] = cv2.warpAffine(full, shift, (right - left, bottom - top),
                                       flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return crops, tilt


if __name__ == "__main__":
    # Quick one-board check:  crop_sensors.py <scan.jpg> <out_dir>
    scan, out_dir = sys.argv[1], sys.argv[2]
    name = os.path.basename(scan)
    board_type = int(name.split("_")[1])
    crops, tilt = crop_sensors(scan, board_type)
    print(f"{name}: measured tilt {tilt:+.2f} deg")
    # Did leveling work? Level a small copy the same way and measure the tilt again.
    small = cv2.imread(scan, cv2.IMREAD_REDUCED_COLOR_8)
    sh, sw = small.shape[:2]
    m, _ = level_matrix(small, sw * REDUCE, sh * REDUCE)
    ms = m.copy(); ms[:, 2] /= REDUCE
    leveled = cv2.warpAffine(small, ms, (sw, sh), borderValue=(255, 255, 255))
    cx, cy, residual, size_after = board_outline(leveled)
    print(f"after leveling: tilt {residual:+.2f} deg, centre ({cx/sw:.3f}, {cy/sh:.3f}) of the scan, "
          f"size off by {100 * (size_after - 1):+.1f}%")
    for s, c in crops.items():
        cv2.imwrite(os.path.join(out_dir, f"{name[:-4]}_{s}_crop.png"), c)
        print(f"  {s}: {c.shape[1]} x {c.shape[0]} px")
    cv2.imwrite(os.path.join(out_dir, f"{name[:-4]}_leveled_small.png"), leveled)

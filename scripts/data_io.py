"""Read data/inputs/{train,eval}.safetensors into plain arrays for a model.

What this does, in plain words
------------------------------
The two files hold one block of numbers per sensor (U1, U2, U3, U4) because the four
sensors have different picture sizes.  This script stacks the four sensors into ONE list
of samples (U1 first, then U2, ...) and, if asked, shrinks every picture to a small square
so one model can look at all of them.

What comes back (N = number of samples), as a dictionary:
  y_ms             time to failure in milliseconds (the answer we want to guess)
  sensor           0, 1, 2, 3 for U1, U2, U3, U4
  fold             0-4 cross-validation fold (training file), or -1 (evaluation file)
  board            board ID of each sample, e.g. "03_01_0031"
  last_current_ms  the last time stamp of the sample's current-vs-time curve, or NaN if the
                   sample has no current file.  (This equals the time to failure - a label
                   leak.  It is used ONLY for the positive control, never by the model.)
  x                only if `pixels` is given: pictures, float32 [N, 2, pixels, pixels],
                   0-1 scale, layer 0 = pristine, layer 1 = exposed

Pictures are squeezed to a square without keeping their shape (U4 is long and thin).  That
throws detail away, but it is simple, and a better model can do its own loading in Part II.
"""
import json

import cv2
import numpy as np

from make_tensors import SENSORS


def shrink(pictures, pixels):
    """uint8 [N, 2, H, W] -> float32 [N, 2, pixels, pixels] on a 0-1 scale (area average)."""
    out = np.empty((len(pictures), 2, pixels, pixels), np.float32)
    for i in range(len(pictures)):
        for layer in range(2):
            small = cv2.resize(pictures[i, layer], (pixels, pixels), interpolation=cv2.INTER_AREA)
            out[i, layer] = small / 255.0
    return out


def last_time_stamp(current, length):
    """Last real time stamp of each current curve (NaN when the sample has no current file)."""
    n = len(length)
    last = current[np.arange(n), 0, np.maximum(length - 1, 0)].astype(np.float64)
    return np.where(length > 0, last, np.nan)


def load_split(path, pixels=None):
    """Read one safetensors file (see the top of this file for what comes back)."""
    from safetensors import safe_open
    pieces = {"y_ms": [], "sensor": [], "fold": [], "board": [], "last_current_ms": [], "x": []}
    with safe_open(path, framework="np") as f:
        notes = f.metadata()
        board_names = json.loads(notes["boards"])
        for k, sensor in enumerate(SENSORS):
            y = f.get_tensor(f"{sensor}_y")
            pieces["y_ms"].append(y.astype(np.float64))
            pieces["sensor"].append(np.full(len(y), k, np.int64))
            pieces["fold"].append(f.get_tensor(f"{sensor}_fold"))
            pieces["board"].append(np.array([board_names[b] for b in f.get_tensor(f"{sensor}_board")]))
            pieces["last_current_ms"].append(
                last_time_stamp(f.get_tensor(f"{sensor}_current"), f.get_tensor(f"{sensor}_curve_len")[:, 0]))
            if pixels:
                pieces["x"].append(shrink(f.get_tensor(f"{sensor}_x"), pixels))   # one sensor at a time: low memory
    out = {k: np.concatenate(v) for k, v in pieces.items() if v}
    out["scale"] = float(notes["scale"])
    return out

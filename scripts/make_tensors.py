"""Build data/inputs/train.safetensors and data/inputs/eval.safetensors.

What this does, in plain words
------------------------------
For every sample (one sensor on one board) we make a two-layer picture:
layer 0 = the pristine (unused) sensor, layer 1 = the exposed (tested) sensor.
Both are grey-scale and cut to the same fixed size for that sensor (U1, U2, U3 or U4
each have their own shape), with the pristine enlarged to match the exposed scan.
The label is the time to failure in milliseconds.

Every board goes entirely to training or entirely to evaluation (see split.py).
The training file also records which of the 5 cross-validation folds each sample is in.

Names inside a file, for each sensor S in U1..U4  (N = number of samples):
  S_x      uint8   [N, 2, H, W]   pristine + exposed pictures
  S_y      float32 [N]            time to failure, ms
  S_fold   int64   [N]            training fold 0-4, or -1 for evaluation
  S_board  int64   [N]            which board (position in the "boards" list in the file notes)
  S_run    int64   [N]            1 = first test of that sensor, 2 = second test, ...
  S_current, S_cv_exposed, S_cv_pristine, S_cf_exposed, S_cf_pristine
           float32 [N, columns, points]   the full curves, NaN-padded (see curve_arrays.py)
  S_curve_len  int64 [N, 5]       real number of points in each of those five curves (0 = none)

Size setting: IDC_SCALE (default 1 = full resolution). IDC_SCALE=0.25 makes a
quick, small version for testing.
"""
import json
import os
import sys

import cv2
import numpy as np

from crop_sensors import crop_sensors
from curve_arrays import SET_NAMES, describe_curves, load_curves, empty_slots, fill_slot, max_lengths
from samples import check_samples, load_samples
from split import N_FOLDS, SEED, assign_boards, check_no_leakage

SENSORS = ("U1", "U2", "U3", "U4")
# Fixed picture size (width, height) per sensor at full resolution; multiples of 16.
FULL_SIZE = {"U1": (2496, 1136), "U2": (1408, 1840), "U3": (1232, 2768), "U4": (1248, 3696)}
OUT_DIR = "data/inputs"


def target_size(sensor, scale):
    """Width and height for this sensor, rounded to a multiple of 16."""
    w, h = FULL_SIZE[sensor]
    return max(16, round(w * scale / 16) * 16), max(16, round(h * scale / 16) * 16)


def fixed_grey(bgr_crop, size):
    """Grey-scale copy of a crop, resized to exactly `size` = (width, height)."""
    grey = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    shrinking = grey.shape[0] * grey.shape[1] > size[0] * size[1]
    return cv2.resize(grey, size, interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC)


def build_tensors(samples, roles, scale=1.0, repo_root=".", curves=None):
    """Return {"train": (tensors, notes), "eval": (tensors, notes)}.

    Pictures are cut from each board's scan ONCE and copied into place, so memory
    stays close to the size of the final files.
    """
    split_of = lambda s: "eval" if roles[s["board_id"]] == -1 else "train"
    # Where each sample lands: its position inside its split's array for its sensor.
    slot, counts = {}, {}
    for i, s in enumerate(samples):
        key = (split_of(s), s["sensor"])
        slot[i] = counts.get(key, 0)
        counts[key] = slot[i] + 1
    tensors = {}
    for (sp, sensor), n in counts.items():
        w, h = target_size(sensor, scale)
        tensors[sp, sensor] = dict(
            x=np.zeros((n, 2, h, w), np.uint8), y=np.zeros(n, np.float32),
            fold=np.zeros(n, np.int64), board=np.zeros(n, np.int64), run=np.zeros(n, np.int64))
    longest = max_lengths(curves, samples)    # one width per curve set, shared by train and eval
    for (sp, sensor), n in counts.items():
        tensors[sp, sensor].update(empty_slots(n, longest))
    boards = {sp: sorted({s["board_id"] for s in samples if split_of(s) == sp}) for sp in ("train", "eval")}

    pristine = {}  # pristine crops for each board type, made once
    by_board = {}
    for i, s in enumerate(samples):
        by_board.setdefault(s["board_id"], []).append(i)
    for board_id in sorted(by_board):
        first = samples[by_board[board_id][0]]
        exposed, _ = crop_sensors(os.path.join(repo_root, first["exposed_path"]), first["board_type"])
        if first["board_type"] not in pristine:
            crops, _ = crop_sensors(os.path.join(repo_root, first["pristine_path"]), first["board_type"])
            pristine[first["board_type"]] = crops
        for i in by_board[board_id]:
            s = samples[i]
            sp, sensor = split_of(s), s["sensor"]
            size = target_size(sensor, scale)
            t, j = tensors[sp, sensor], slot[i]
            t["x"][j, 0] = fixed_grey(pristine[s["board_type"]][sensor], size)
            t["x"][j, 1] = fixed_grey(exposed[sensor], size)
            t["y"][j] = s["ttf_ms"]
            t["fold"][j] = roles[board_id]
            t["board"][j] = boards[sp].index(board_id)
            t["run"][j] = s["run_number"]
            fill_slot(t, j, (curves or {}).get((board_id, sensor, s["run_number"]), {}))
        print(f"  {board_id}: {len(by_board[board_id])} sensors", flush=True)

    out = {}
    for sp in ("train", "eval"):
        flat = {f"{sensor}_{name}": arr for (p, sensor), d in tensors.items() if p == sp for name, arr in d.items()}
        notes = {"boards": json.dumps(boards[sp]), "seed": str(SEED), "n_folds": str(N_FOLDS),
                 "scale": str(scale), "layers": "0=pristine,1=exposed (grey-scale)", "label": "time to failure, ms",
                 "curves": describe_curves()}
        out[sp] = (flat, notes)
    return out


def main():
    import duckdb
    from safetensors.numpy import save_file
    scale = float(os.environ.get("IDC_SCALE", "1"))
    con = duckdb.connect("build/idc.duckdb", read_only=True)
    samples = load_samples(con)
    curves = load_curves(con)
    check_samples(samples)
    roles = assign_boards({s["board_id"]: s["board_type"] for s in samples})
    check_no_leakage(samples, roles)
    print(f"building {len(samples)} samples at scale {scale}")
    os.makedirs(OUT_DIR, exist_ok=True)
    for sp, (flat, notes) in build_tensors(samples, roles, scale, curves=curves).items():
        final = os.path.join(OUT_DIR, f"{sp}.safetensors")
        save_file(flat, final + ".tmp", metadata=notes)   # write aside first ...
        os.replace(final + ".tmp", final)                 # ... then swap in: never a half-written file
        print(f"wrote {final} ({os.path.getsize(final) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()

"""Check the finished train/eval files. Run with:  uv run python scripts/check.py

What this does, in plain words
------------------------------
1. Opens data/inputs/train.safetensors and eval.safetensors the way a model would
   (safetensors.torch.load_file -> TensorDataset -> DataLoader).
2. Checks the pictures have the right shape and are not blank, and the labels are sane.
3. Checks that no board appears in both training and evaluation (no leakage),
   and that every cross-validation fold is non-empty and never mixes boards with its
   own training part.
Stops with a clear "STOP: ..." message at the first problem; prints CHECK OK if all is well.
"""
import json
import sys

import numpy as np

from make_tensors import SENSORS, target_size
from split import N_FOLDS


def fail(message):
    sys.exit(f"STOP: {message}")


def check_arrays(files, notes):
    """files / notes: {"train": {...}, "eval": {...}} of numpy arrays / text notes."""
    boards = {sp: json.loads(notes[sp]["boards"]) for sp in ("train", "eval")}
    # 1. No board on both sides.
    both = set(boards["train"]) & set(boards["eval"])
    if both:
        fail(f"boards in both train and eval: {sorted(both)}")
    for sp in ("train", "eval"):
        scale = float(notes[sp]["scale"])
        for sensor in SENSORS:
            for name in ("x", "y", "fold", "board", "run"):
                if f"{sensor}_{name}" not in files[sp]:
                    fail(f"{sp} file has no {sensor}_{name}")
            x, y = files[sp][f"{sensor}_x"], files[sp][f"{sensor}_y"]
            fold, board = files[sp][f"{sensor}_fold"], files[sp][f"{sensor}_board"]
            w, h = target_size(sensor, scale)
            # 2. Shapes, blank pictures, labels.
            if x.shape[1:] != (2, h, w):
                fail(f"{sp} {sensor}_x has shape {x.shape}, expected (N, 2, {h}, {w})")
            if not (len(x) == len(y) == len(fold) == len(board)):
                fail(f"{sp} {sensor}: pictures, labels and folds have different lengths")
            if (x.min(axis=(2, 3)) == x.max(axis=(2, 3))).any():   # blank = one grey level only (no big copies)
                fail(f"{sp} {sensor}: a picture layer is completely blank")
            if not np.isfinite(y).all() or (y <= 0).any():
                fail(f"{sp} {sensor}: a time-to-failure is missing or not positive")
            # Which side of the split each sample is on must match the file it is in.
            if sp == "eval" and (fold != -1).any():
                fail(f"eval file has a training sample ({sensor})")
            if sp == "train" and (fold < 0).any():
                fail(f"train file has an evaluation sample ({sensor})")
            if board.min() < 0 or board.max() >= len(boards[sp]):
                fail(f"{sp} {sensor}: board number out of range")
    # 3. Cross-validation folds: non-empty, and a fold's boards never appear in the rest.
    for f in range(N_FOLDS):
        held_out, rest = set(), set()
        for sensor in SENSORS:
            fold, board = files["train"][f"{sensor}_fold"], files["train"][f"{sensor}_board"]
            held_out |= {boards["train"][b] for b in board[fold == f]}
            rest |= {boards["train"][b] for b in board[fold != f]}
        if not held_out:
            fail(f"fold {f} is empty")
        if held_out & rest:
            fail(f"fold {f} shares boards with the other folds: {sorted(held_out & rest)[:3]}")
    return boards


def main():
    from safetensors import safe_open
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader, TensorDataset

    tensors, files, notes = {}, {}, {}
    for sp in ("train", "eval"):
        path = f"data/inputs/{sp}.safetensors"
        tensors[sp] = load_file(path)                       # the line a model would use
        files[sp] = {k: v.numpy() for k, v in tensors[sp].items()}
        with safe_open(path, framework="np") as fh:
            notes[sp] = fh.metadata()
    boards = check_arrays(files, notes)
    for sp in ("train", "eval"):
        print(f"{sp}: {len(boards[sp])} boards;",
              ", ".join(f"{s} {len(files[sp][s + '_y'])}" for s in SENSORS), "samples")

    # A real DataLoader on one sensor, one cross-validation fold held out.
    t = tensors["train"]
    keep = t["U1_fold"] != 0
    data = TensorDataset(t["U1_x"][keep], t["U1_y"][keep])
    x, y = next(iter(DataLoader(data, batch_size=4, shuffle=False)))
    print(f"DataLoader batch: pictures {tuple(x.shape)} {x.dtype}, labels {tuple(y.shape)} {y.dtype}")
    print("CHECK OK")


if __name__ == "__main__":
    main()

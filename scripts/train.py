"""Train the model on the training file and save it.  Run with:  uv run python scripts/train.py

What this does, in plain words
------------------------------
1. Reads ONLY data/inputs/train.safetensors.  The held-out evaluation file is never opened
   here (the Makefile does not even list it as an input), so training cannot peek at it.
2. Sets one cross-validation fold aside (VAL_FOLD, whole boards) as a "check set".  The model
   learns from the other folds and is scored on the check set after every epoch, so we can
   see whether it is really learning or just memorising.
3. Prints one line per epoch, then a small text picture of the training curve.
4. Keeps the weights from the epoch with the best check-set score ("early stopping") and
   writes them to build/model/model.npz, plus the curve to build/model/history.json.

Loss numbers are the average squared mistake in "standardised" units: 1.0 means no better
than always guessing the training average; below 1.0 means the model has learned something.
"""
import json
import os
import time

import numpy as np

from data_io import load_split
from model import (BATCH, EPOCHS, HIDDEN, LEARNING_RATE, PIXELS, SEED, VAL_FOLD, WEIGHT_DECAY,
                   Adam, Baseline, loss_and_grads, forward)

OUT_DIR = "build/model"


def text_curve(history, rows=15, width=40):
    """A small text picture of the check-set loss; '|' marks 1.0 (= guessing the average)."""
    val = np.array(history["val_loss"])
    top = max(1.2, float(val.max()))
    lines = []
    for e in np.linspace(0, len(val) - 1, min(rows, len(val))).round().astype(int):
        bar = ["#" if i < val[e] / top * width else " " for i in range(width)]
        bar[min(width - 1, int(1.0 / top * width))] = "|"
        lines.append(f"  epoch {e + 1:3d} {val[e]:5.2f} {''.join(bar)}")
    return "\n".join(lines)


def main():
    t0 = time.time()
    data = load_split("data/inputs/train.safetensors", pixels=PIXELS)
    y_log = np.log10(data["y_ms"])
    val = data["fold"] == VAL_FOLD
    tr = ~val
    boards_tr, boards_val = set(data["board"][tr]), set(data["board"][val])
    assert not boards_tr & boards_val, "a board is in both the training part and the check set"
    print(f"training file: {len(y_log)} samples ({tr.sum()} to learn from, {val.sum()} check set = fold {VAL_FOLD}), "
          f"{len(boards_tr)} + {len(boards_val)} boards, pictures {PIXELS}x{PIXELS}, "
          f"loaded in {time.time() - t0:.0f}s", flush=True)

    model = Baseline()
    model.fit_scalers(data["x"][tr], data["sensor"][tr], y_log[tr])      # rescaling comes from training samples only
    X_tr = model.scaled_inputs(data["x"][tr], data["sensor"][tr])
    X_val = model.scaled_inputs(data["x"][val], data["sensor"][val])
    y_tr, y_val = model.scaled_answers(y_log[tr]), model.scaled_answers(y_log[val])
    rng = np.random.default_rng(SEED)
    model.init_weights(X_tr.shape[1], rng)
    adam = Adam(model.params, LEARNING_RATE)
    print(f"model: {X_tr.shape[1]} inputs -> {HIDDEN} hidden -> 1 output ({model.n_parameters()} weights); "
          f"{EPOCHS} epochs, batch {BATCH}, learning rate {LEARNING_RATE}, weight decay {WEIGHT_DECAY}", flush=True)

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best, best_params, step = (np.inf, 0), None, 0
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(y_tr))
        for start in range(0, len(order), BATCH):
            rows = order[start:start + BATCH]
            _, grads = loss_and_grads(model.params, X_tr[rows], y_tr[rows], WEIGHT_DECAY)
            adam.step(model.params, grads)
            step += 1
        train_loss = float(np.mean((forward(model.params, X_tr)[0] - y_tr) ** 2))
        val_loss = float(np.mean((forward(model.params, X_val)[0] - y_val) ** 2))
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best[0]:
            best, best_params = (val_loss, epoch), {k: v.copy() for k, v in model.params.items()}
        print(f"epoch {epoch:3d}/{EPOCHS}  step {step:4d}  train loss {train_loss:6.3f}  check-set loss {val_loss:6.3f}"
              f"{'  <- best so far' if best[1] == epoch else ''}", flush=True)

    model.params = best_params                                           # early stopping: go back to the best epoch
    dummy_val_loss = float(np.mean(y_val ** 2))                          # the guess-the-average model on the check set
    history.update(best_epoch=best[1], best_val_loss=best[0], dummy_val_loss=dummy_val_loss, steps=step,
                   n_train=int(tr.sum()), n_val=int(val.sum()), val_fold=VAL_FOLD, pixels=PIXELS, hidden=HIDDEN,
                   epochs=EPOCHS, batch=BATCH, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                   n_weights=model.n_parameters(), seed=SEED, scale=data["scale"])
    print("\ncheck-set loss by epoch ('|' = 1.0, the guess-the-average level):")
    print(text_curve(history))
    print(f"best epoch {best[1]}: check-set loss {best[0]:.3f} (guess-the-average: {dummy_val_loss:.3f})")

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save(f"{OUT_DIR}/model.tmp.npz")
    with open(f"{OUT_DIR}/history.tmp.json", "w") as fh:
        json.dump(history, fh)
    os.replace(f"{OUT_DIR}/model.tmp.npz", f"{OUT_DIR}/model.npz")       # swap in last: never a half-written file
    os.replace(f"{OUT_DIR}/history.tmp.json", f"{OUT_DIR}/history.json")
    print(f"wrote {OUT_DIR}/model.npz and {OUT_DIR}/history.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

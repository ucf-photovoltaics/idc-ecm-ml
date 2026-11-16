"""Score the trained model on the HELD-OUT file, next to a negative and a positive control.
Run with:  uv run python scripts/evaluate.py      Writes results/results.json.

What this does, in plain words
------------------------------
The held-out file (data/inputs/eval.safetensors) holds boards the model never saw while
training.  Here we ask three "models" to guess log10(time to failure) for those sensors:

  model              the neural network trained by train.py.
  dummy              NEGATIVE control: ignores the pictures and always guesses the training
                     average.  Any real model must beat this.  It scores R2 <= 0 and AUC 0.5
                     by construction; if it ever does not, the scoring code is broken.
  positive_control   POSITIVE control: cheats on purpose.  It reads the last time stamp of the
                     sensor's current-vs-time curve, which IS the time to failure (a label
                     leak, see curve_arrays.py), and fits a straight line to it using the
                     TRAINING samples.  It must score very well; if it does not, the
                     pipeline (loading, ordering, scoring) is broken.  Sensors with no current
                     file get the training average, so it is not perfect.

Numbers reported for each (all on the held-out file):
  R2, MAE, RMSE, Spearman   how close the guesses are (log10 ms; see metrics.py)
  AUC                       can the guess tell short-lived sensors (below the training MEDIAN
                            time to failure) from long-lived ones?  0.5 = coin flip.
  95% intervals             re-drawing whole boards 1000 times (board_bootstrap_ci).

Everything the report needs goes into results/results.json; figures and the report are made
from that file alone.
"""
import json
import os
import re
import sys

import numpy as np

import metrics as K
from data_io import load_split
from model import PIXELS, SEED, Baseline

TRAIN, EVAL = "data/inputs/train.safetensors", "data/inputs/eval.safetensors"
OUT = "results/results.json"
N_BOOT = 1000


def fit_positive_control(train, y_train):
    """Straight line  log10(time to failure) = slope * log10(last current time stamp) + intercept."""
    usable = np.isfinite(train["last_current_ms"]) & (train["last_current_ms"] > 0)
    slope, intercept = np.polyfit(np.log10(train["last_current_ms"][usable]), y_train[usable], 1)
    return float(slope), float(intercept), int(usable.sum())


def positive_control_guesses(test, slope, intercept, fallback):
    usable = np.isfinite(test["last_current_ms"]) & (test["last_current_ms"] > 0)
    guesses = np.full(len(usable), fallback)
    guesses[usable] = slope * np.log10(test["last_current_ms"][usable]) + intercept
    return guesses, usable


def all_scores(y, p, short):
    """Every number we report, for one set of guesses."""
    return {"r2": K.r2(y, p), "mae": K.mae(y, p), "rmse": K.rmse(y, p), "spearman": K.spearman(y, p),
            "auc": K.auc(short, -p)}                       # lower guessed time = more likely short-lived


def plain(obj):
    """Turn numpy values into plain JSON values; NaN/inf become null."""
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [plain(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 6) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    return obj


def dump(obj, path):
    """Write JSON with short lists on one line (so the file stays readable); swap in at the end."""
    text = json.dumps(plain(obj), indent=2)
    text = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", text)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        fh.write(text + "\n")
    os.replace(path + ".tmp", path)


def main():
    train = load_split(TRAIN)                            # no pictures needed: only labels and the leaky time stamp
    test = load_split(EVAL, pixels=PIXELS)
    overlap = sorted(set(train["board"]) & set(test["board"]))
    if overlap:
        sys.exit(f"STOP: boards in both the training and the held-out file: {overlap[:3]}")

    model = Baseline.load("build/model/model.npz")
    with open("build/model/history.json") as fh:
        history = json.load(fh)
    y_train, y_true = np.log10(train["y_ms"]), np.log10(test["y_ms"])
    threshold = float(np.median(y_train))                # "short-lived" = below the TRAINING median
    short = y_true < threshold

    slope, intercept, n_fit = fit_positive_control(train, y_train)
    pos_guess, has_current = positive_control_guesses(test, slope, intercept, float(y_train.mean()))
    guesses = {
        "model": (model.predict(test["x"], test["sensor"]),
                  "Neural network (trained on the pictures)"),
        "dummy": (np.full(len(y_true), float(y_train.mean())),
                  "Dummy: always the training average (negative control)"),
        "positive_control": (pos_guess,
                             "Last current time stamp (positive control; leaks the answer on purpose)"),
    }

    models = {}
    for name, (p, label) in guesses.items():
        fpr, tpr = K.roc_curve(short, -p)
        ci = K.board_bootstrap_ci(
            test["board"], lambda rows, p=p: {k: v for k, v in all_scores(y_true[rows], p[rows], short[rows]).items()
                                              if k in ("r2", "mae", "auc")},
            n_boot=N_BOOT, seed=SEED)
        scores = all_scores(y_true, p, short)
        scores["fold_error"] = 10 ** scores["mae"]       # typical miss as a factor, e.g. 2.0 = off by 2x
        # Same scores, but only on sensors that HAVE a current file: the fair place to judge the positive control.
        on_subset = all_scores(y_true[has_current], p[has_current], short[has_current])
        models[name] = {"label": label, "y_pred": p, "metrics": scores, "ci95": ci,
                        "metrics_current_subset": {k: on_subset[k] for k in ("r2", "mae", "auc")},
                        "roc": {"fpr": fpr, "tpr": tpr}}
    models["positive_control"]["n_with_current_file"] = int(has_current.sum())
    models["positive_control"]["fit"] = {"slope": slope, "intercept": intercept, "n_training_samples": n_fit}

    results = {
        "schema": 1,
        "target": "log10(time to failure, ms) of one sensor, from its pristine + exposed pictures",
        "data": {
            "n_train_file": int(len(train["y_ms"])), "n_check_set": history["n_val"],
            "n_learned_from": history["n_train"], "n_test": int(len(y_true)),
            "n_boards_train_file": int(len(set(train["board"]))), "n_boards_test": int(len(set(test["board"]))),
            "boards_in_both": len(overlap), "scale": test["scale"], "pixels": PIXELS, "seed": SEED,
            "n_bootstrap": N_BOOT,
            "test_per_sensor": {s: int((test["sensor"] == k).sum()) for k, s in enumerate(("U1", "U2", "U3", "U4"))},
        },
        "training": history,
        "test": {
            "board": test["board"], "sensor": [("U1", "U2", "U3", "U4")[k] for k in test["sensor"]],
            "y_true": y_true, "threshold_log10_ms": threshold,
            "n_short_lived": int(short.sum()), "models": models,
        },
    }
    dump(results, OUT)

    print(f"held-out file: {len(y_true)} sensors from {len(set(test['board']))} boards "
          f"(no board shared with training); short-lived = below {10 ** threshold / 1000:.0f} s, "
          f"{int(short.sum())} of {len(y_true)} sensors")
    print(f"{'':18s} {'R2':>7s} {'MAE':>7s} {'Spearman':>9s} {'AUC':>7s}   (95% interval of R2 and AUC)")
    for name, m in models.items():
        s, ci = m["metrics"], m["ci95"]
        f = lambda v: "   n/a " if v is None else f"{v:7.3f}"
        rng = lambda k: "[%.2f, %.2f]" % tuple(ci[k]) if k in ci else "[n/a]"
        print(f"{name:18s} {f(s['r2'])} {f(s['mae'])} {f(s['spearman'])}  {f(s['auc'])}   R2 {rng('r2')}  AUC {rng('auc')}")
    print(f"only the {int(has_current.sum())} of {len(y_true)} held-out sensors that have a current file:")
    for name, m in models.items():
        s = m["metrics_current_subset"]
        print(f"{name:18s} R2 {s['r2']:7.3f}   MAE {s['mae']:6.3f}   AUC {s['auc']:6.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

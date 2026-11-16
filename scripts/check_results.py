"""Check results/results.json, the figures and the report.  Run with:  uv run python scripts/check_results.py

What this does, in plain words
------------------------------
Training a model that "runs" is not the same as a pipeline you can trust.  This script stops with a
clear "STOP: ..." message if any of these is not true:

 1. results.json has every part the report needs, and every number is finite.
 2. No board is in both the training file and the held-out file, and the test set is not empty.
 3. NEGATIVE control: the dummy scores AUC 0.5 and R2 <= 0 (it cannot learn, so anything else
    means the scoring code is wrong).
 4. POSITIVE control: on held-out sensors that have a current file, the "cheating" model reproduces
    the truth (R2 >= 0.9, AUC >= 0.95).  If it does not, loading, ordering or scoring is broken.
 5. Training happened: the training loss at the end is lower than at the start, and the saved best
    epoch exists.
 6. The figures and report.pdf exist, and the numbers in the report (build/report/numbers.tex)
    are the numbers in results.json.

Prints CHECK OK if all is well.  It does NOT require the model to be good - only that the machinery works.
"""
import json
import math
import os
import re
import sys

RESULTS = "results/results.json"
FILES = ["results/figures/training_curve.pdf", "results/figures/roc.pdf", "results/figures/parity.pdf",
         "results/report.pdf"]


def fail(message):
    sys.exit(f"STOP: {message}")


def finite(values, what):
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        fail(f"{what} has a missing or non-finite number")


def main():
    if not os.path.exists(RESULTS):
        fail(f"{RESULTS} is missing (run `make all` first)")
    with open(RESULTS) as fh:
        r = json.load(fh)
    d, tr, t = r["data"], r["training"], r["test"]
    models = t["models"]

    # 1. Complete and finite.
    for name in ("model", "dummy", "positive_control"):
        if name not in models:
            fail(f"results.json has no '{name}'")
        m = models[name]
        if len(m["y_pred"]) != d["n_test"] or len(t["y_true"]) != d["n_test"]:
            fail(f"{name}: {len(m['y_pred'])} guesses for {d['n_test']} held-out samples")
        finite(m["y_pred"], f"{name} guesses")
        for key in ("r2", "mae", "rmse", "auc"):
            if m["metrics"][key] is None:
                fail(f"{name}: {key} could not be computed")
        if len(m["roc"]["fpr"]) != len(m["roc"]["tpr"]) or m["roc"]["fpr"][0] != 0 or m["roc"]["fpr"][-1] != 1:
            fail(f"{name}: the ROC curve is malformed")
    finite(t["y_true"], "held-out true values")

    # 2. No leakage, non-empty test set.
    if d["boards_in_both"] != 0:
        fail(f"{d['boards_in_both']} boards are in both the training and the held-out file")
    if d["n_test"] < 10 or t["n_short_lived"] in (0, d["n_test"]):
        fail("the held-out set is too small, or has only one kind of sensor, to score")

    # 3. Negative control.
    dummy = models["dummy"]["metrics"]
    if abs(dummy["auc"] - 0.5) > 1e-9:
        fail(f"the dummy has AUC {dummy['auc']:.4f}, not 0.5: the scoring is wrong")
    if dummy["r2"] > 1e-6:
        fail(f"the dummy has R2 {dummy['r2']:.4f} > 0: the scoring is wrong")

    # 4. Positive control, on the sensors where it has something to read.
    pos, n_cur = models["positive_control"]["metrics_current_subset"], models["positive_control"]["n_with_current_file"]
    if n_cur < 5:
        fail(f"only {n_cur} held-out sensors have a current file: the positive control cannot be checked")
    if pos["r2"] is None or pos["r2"] < 0.9 or (pos["auc"] is not None and pos["auc"] < 0.95):
        fail(f"the positive control does not reproduce the truth (R2 {pos['r2']}, AUC {pos['auc']}): "
             "loading, ordering or scoring is broken")

    # 5. Training happened.
    n = len(tr["epoch"])
    if n != tr["epochs"] or len(tr["train_loss"]) != n or len(tr["val_loss"]) != n:
        fail("the training history has the wrong length")
    finite(tr["train_loss"] + tr["val_loss"], "the training history")
    if not 1 <= tr["best_epoch"] <= n:
        fail("best_epoch is outside the training history")
    if not tr["train_loss"][-1] < tr["train_loss"][0]:
        fail("the training loss did not go down: the model is not learning")

    # 6. Figures, report, and a report that matches results.json.
    for path in FILES:
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            fail(f"{path} is missing or empty")
        with open(path, "rb") as fh:
            if fh.read(5) != b"%PDF-":
                fail(f"{path} is not a PDF")
    with open("build/report/numbers.tex") as fh:
        macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^\n]*)\}$", fh.read(), flags=re.M))
    expect = {"NTest": str(d["n_test"]), "NBoardsTest": str(d["n_boards_test"]), "BestEpoch": str(tr["best_epoch"]),
              "AucModel": f"{models['model']['metrics']['auc']:.2f}", "AucPos": f"{models['positive_control']['metrics']['auc']:.2f}"}
    for key, value in expect.items():
        if macros.get(key) != value:
            fail(f"the report says \\{key} = {macros.get(key)}, but results.json says {value} (report is out of date)")
    if os.path.getmtime("results/report.pdf") < os.path.getmtime(RESULTS):
        fail("results/report.pdf is older than results/results.json")

    m = models["model"]["metrics"]
    print(f"held-out: {d['n_test']} sensors, {d['n_boards_test']} boards, none shared with training")
    print(f"negative control: dummy AUC {dummy['auc']:.2f}, R2 {dummy['r2']:.2f}")
    print(f"positive control: R2 {pos['r2']:.3f}, AUC {pos['auc']:.2f} on the {n_cur} sensors with a current file")
    print(f"model: R2 {m['r2']:.2f}, AUC {m['auc']:.2f} (a weak baseline is fine for Part I)")
    print("CHECK OK")


if __name__ == "__main__":
    main()

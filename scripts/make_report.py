"""Turn results/results.json into LaTeX "number macros".  Run with:  uv run python scripts/make_report.py

What this does, in plain words
------------------------------
The report (report/report.tex) never contains a typed-in result.  Instead it uses names like
\\AucModel or \\NTest, and THIS script writes build/report/numbers.tex, which says what each
name stands for, using the current results/results.json.  So when the data change and the
analysis re-runs, the report text and tables update by themselves.

Names are letters only (LaTeX does not allow digits in names).  Pattern:
    \\Rsq / \\Mae / \\Rho / \\Auc / \\Fold  +  Model | Dummy | Pos  (+ Lo / Hi for a 95% interval)
    \\RsqSub...  the same, scored only on sensors that have a current file
plus counts such as \\NTest, \\NBoardsTest, \\BestEpoch.  \\metricstable and \\subsettable are
ready-made tables.
"""
import json
import os

RESULTS, OUT = "results/results.json", "build/report/numbers.tex"
SUFFIX = {"model": "Model", "dummy": "Dummy", "positive_control": "Pos"}
NAMES = {"model": "model (neural network)", "dummy": "dummy (negative control)",
         "positive_control": "current time stamp (positive control)"}


def tex_escape(s):
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("$", "\\$")):
        s = s.replace(a, b)
    return s


def num(v, digits=2):
    """A number as LaTeX text; negatives get a real minus sign; None becomes n/a."""
    if v is None:
        return "n/a"
    text = f"{abs(v):.{digits}f}"
    return f"$-${text}" if v < 0 and float(text) != 0 else text


def interval(ci, key, digits=2):
    if key not in ci:
        return "n/a"
    lo, hi = ci[key]
    return f"[{num(lo, digits)}, {num(hi, digits)}]"


def define(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main():
    with open(RESULTS) as fh:
        r = json.load(fh)
    d, tr, t = r["data"], r["training"], r["test"]
    lines = ["% Written by scripts/make_report.py from results/results.json - do not edit by hand."]
    add = lambda name, value: lines.append(define(name, value))

    for name, value in {
        "NTrainFile": d["n_train_file"], "NLearned": d["n_learned_from"], "NCheck": d["n_check_set"],
        "NTest": d["n_test"], "NBoardsTrain": d["n_boards_train_file"], "NBoardsTest": d["n_boards_test"],
        "NBoardsShared": d["boards_in_both"], "Scale": d["scale"], "Pixels": d["pixels"], "Seed": d["seed"],
        "NBoot": d["n_bootstrap"], "NShort": t["n_short_lived"],
        "ThresholdSeconds": f"{10 ** t['threshold_log10_ms'] / 1000:.0f}",
        "NWithCurrent": t["models"]["positive_control"]["n_with_current_file"],
        "BestEpoch": tr["best_epoch"], "Epochs": tr["epochs"], "NSteps": tr["steps"], "Batch": tr["batch"],
        "Hidden": tr["hidden"], "NWeights": tr["n_weights"], "ValFold": tr["val_fold"],
        "LearningRate": tr["learning_rate"], "WeightDecay": tr["weight_decay"],
        "BestValLoss": num(tr["best_val_loss"], 2), "DummyValLoss": num(tr["dummy_val_loss"], 2),
        "FinalTrainLoss": num(tr["train_loss"][-1], 2),
        "PerSensor": ", ".join(f"{k}: {v}" for k, v in d["test_per_sensor"].items()),
    }.items():
        add(name, value)

    for key, suffix in SUFFIX.items():
        m, ci, sub = t["models"][key]["metrics"], t["models"][key]["ci95"], t["models"][key]["metrics_current_subset"]
        add("Rsq" + suffix, num(m["r2"]))
        add("Mae" + suffix, num(m["mae"]))
        add("Rho" + suffix, num(m["spearman"]))
        add("Auc" + suffix, num(m["auc"]))
        add("Fold" + suffix, f"{m['fold_error']:.1f}$\\times$")
        add("RsqInt" + suffix, interval(ci, "r2"))
        add("AucInt" + suffix, interval(ci, "auc"))
        add("RsqSub" + suffix, num(sub["r2"]))
        add("MaeSub" + suffix, num(sub["mae"]))
        add("AucSub" + suffix, num(sub["auc"]))

    auc_ci = t["models"]["model"]["ci95"].get("auc")
    if auc_ci is None:
        verdict = "The interval for the model's AUC could not be computed."
    elif auc_ci[0] > 0.5:
        verdict = "The 95\\% interval for the model's AUC lies above 0.5 (the dummy's value)."
    else:
        verdict = "The 95\\% interval for the model's AUC includes 0.5 (the dummy's value), so this run cannot show the model is better than guessing."
    add("AucVerdict", verdict)

    rows = [f"{tex_escape(NAMES[k])} & {num(t['models'][k]['metrics']['r2'])} {interval(t['models'][k]['ci95'], 'r2')} & "
            f"{t['models'][k]['metrics']['fold_error']:.1f}$\\times$ & "
            f"{num(t['models'][k]['metrics']['auc'])} {interval(t['models'][k]['ci95'], 'auc')} \\\\" for k in SUFFIX]
    add("metricstable", "\\begin{tabular}{lccc}\\toprule\n & $R^2$ [95\\% interval] & typical miss & AUC [95\\% interval] \\\\\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
    rows = [f"{tex_escape(NAMES[k])} & {num(t['models'][k]['metrics_current_subset']['r2'])} & "
            f"{num(t['models'][k]['metrics_current_subset']['mae'], 3)} & {num(t['models'][k]['metrics_current_subset']['auc'])} \\\\"
            for k in SUFFIX]
    add("subsettable", "\\begin{tabular}{lccc}\\toprule\n & $R^2$ & MAE (log$_{10}$ ms) & AUC \\\\\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(OUT + ".tmp", OUT)
    print(f"wrote {OUT} ({len(lines) - 1} macros)")


if __name__ == "__main__":
    main()

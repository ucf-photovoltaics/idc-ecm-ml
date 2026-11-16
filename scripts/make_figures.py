"""Draw the report's figures from results/results.json.  Run with:  uv run python scripts/make_figures.py

What this does, in plain words
------------------------------
Reads ONLY results/results.json (nothing is recomputed here) and writes three PDF pictures
into results/figures/:

  training_curve.pdf   loss after every epoch on the training samples and on the check set,
                       against the guess-the-average level.  Shows whether training works,
                       and where it starts to memorise.
  roc.pdf              ROC curves on the held-out file: how well each model separates short-lived
                       from long-lived sensors.  Diagonal = coin flip; the closer to the top-left
                       corner, the better.  The area under each curve (AUC) is in the legend.
  parity.pdf           guessed vs. true log10 time to failure, one panel per model.  Dots on the
                       diagonal are perfect guesses.

Colours are the same everywhere: model = blue, dummy (negative control) = orange,
positive control = aqua.
"""
import json
import math
import os

from miniplot import AQUA, AXIS, BLUE, INK2, ORANGE, Axes, Page

RESULTS, OUT_DIR = "results/results.json", "results/figures"
LOOK = {"model": ("model", BLUE, None), "dummy": ("dummy (negative control)", ORANGE, (4, 2.5)),
        "positive_control": ("positive control", AQUA, None)}
SHORT = {"model": "model", "dummy": "dummy", "positive_control": "positive control"}   # for narrow panel titles


def f2(v):
    return "n/a" if v is None else f"{v:.2f}"


def training_curve(r, path):
    h = r["training"]
    page = Page(340, 235)
    top = max(1.6, max(h["train_loss"] + h["val_loss"]) * 1.3)      # head-room for the legend
    ax = Axes(page, 48, 38, 280, 170, xlim=(0, h["epochs"]), ylim=(0, round(top, 1)), xlabel="epoch (passes over the training samples)",
              ylabel="average squared mistake (1.0 = guess the average)", title="Training curve", xticks=[0, 10, 20, 30, 40, 50, 60][: h["epochs"] // 10 + 1])
    ax.line([0, h["epochs"]], [h["dummy_val_loss"]] * 2, ORANGE, 1.5, (4, 2.5))
    ax.label(1, h["dummy_val_loss"] - 0.1, "dummy: guess the average (check set)")
    ax.vline(h["best_epoch"])
    ax.label(h["best_epoch"] + 0.8, 0.04, f"best epoch ({h['best_epoch']})")
    ax.line(h["epoch"], h["train_loss"], BLUE, 1.5, (4, 2.5))
    ax.line(h["epoch"], h["val_loss"], BLUE, 1.5)
    ax.legend([("check set (whole boards held back)", BLUE, "line"), ("training samples", BLUE, "dash")], corner="upper right")
    page.save(path)


def roc(r, path):
    t = r["test"]
    page = Page(300, 285)
    ax = Axes(page, 48, 38, 232, 232, xlim=(0, 1), ylim=(0, 1), xlabel="false-positive rate", ylabel="true-positive rate",
              title="ROC on the held-out file: short-lived vs long-lived", xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[0, 0.25, 0.5, 0.75, 1])
    entries = []
    for name in ("dummy", "positive_control", "model"):       # model last, so it is drawn on top
        m = t["models"][name]
        label, color, dash = LOOK[name]
        ax.line(m["roc"]["fpr"], m["roc"]["tpr"], color, 1.5, dash)
        entries.append((f"{label}  AUC {f2(m['metrics']['auc'])}", color, "dash" if dash else "line"))
    ax.legend(entries[::-1], corner="lower right")
    page.save(path)


def parity(r, path):
    t = r["test"]
    values = t["y_true"] + [v for model in t["models"].values() for v in model["y_pred"]]
    lo, hi = math.floor(min(values) * 2) / 2, math.ceil(max(values) * 2) / 2     # round outwards to halves
    page = Page(468, 178)
    panel, gap, left = 118, 32, 40
    for i, name in enumerate(("model", "dummy", "positive_control")):
        m = t["models"][name]
        label, color = SHORT[name], LOOK[name][1]
        ax = Axes(page, left + i * (panel + gap), 38, panel, panel, xlim=(lo, hi), ylim=(lo, hi),
                  xlabel="true log10(time to failure, ms)", ylabel="guessed log10(ms)" if i == 0 else "",
                  title=f"{label}:  R^2 {f2(m['metrics']['r2'])}", xticks=[4, 5, 6], yticks=[4, 5, 6])
        ax.line([lo, hi], [lo, hi], AXIS, 0.75)              # the "perfect guess" diagonal
        ax.scatter(t["y_true"], m["y_pred"], color)
    page.save(path)


def main():
    with open(RESULTS) as fh:
        results = json.load(fh)
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, draw in (("training_curve", training_curve), ("roc", roc), ("parity", parity)):
        draw(results, f"{OUT_DIR}/{name}.pdf")
        print(f"wrote {OUT_DIR}/{name}.pdf")


if __name__ == "__main__":
    main()

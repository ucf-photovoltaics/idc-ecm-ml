"""Scoring helpers: plain numbers in, plain numbers out (no model, no plotting).

What this does, in plain words
------------------------------
Everything here compares a list of TRUE values with a list of GUESSES.

  r2         "How much better than guessing the average?"  1 = perfect, 0 = no better than
             always guessing the test average, below 0 = worse than that.
  mae        average size of the mistake, in the same units as the values (here log10 ms).
  rmse       like mae, but big mistakes count extra.
  spearman   do the guesses put the samples in the right ORDER?  1 = perfect order,
             0 = no relation.  (None when all guesses are identical: order is undefined.)
  roc / auc  turn the problem into "is this sensor short-lived or long-lived?" and ask how
             well the guesses separate the two groups.  0.5 = coin flip, 1.0 = perfect.
  board_bootstrap_ci
             an honest error bar.  Sensors on the same board are not independent, so we
             re-draw whole BOARDS (with replacement) many times and see how much the score
             moves.  The middle 95% of those scores is the "95% interval".
"""
import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    """1 - (squared mistakes of the guess) / (squared mistakes of guessing the average of y)."""
    spread = np.sum((y - np.mean(y)) ** 2)
    if spread == 0:
        return None
    return float(1.0 - np.sum((y - p) ** 2) / spread)


def ranks(a):
    """Rank of each value, 1 = smallest; tied values share the average of their ranks."""
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)                 # position of the last member of each tie group
    first = last - counts + 1                # position of the first member
    return ((first + last) / 2.0)[inverse.reshape(-1)]


def spearman(y, p):
    if np.ptp(y) == 0 or np.ptp(p) == 0:
        return None
    return float(np.corrcoef(ranks(y), ranks(p))[0, 1])


def auc(positive, score):
    """Chance that a random positive sample gets a higher score than a random negative one.

    positive: True/False per sample; score: bigger = "more likely positive".
    Ties count half, so identical scores give exactly 0.5.
    """
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = ranks(np.asarray(score, dtype=float))
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve(positive, score):
    """Points of the ROC curve: (false-positive rate, true-positive rate), from (0,0) to (1,1)."""
    positive = np.asarray(positive, dtype=bool)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")             # highest score first
    pos_sorted, score_sorted = positive[order], score[order]
    tp = np.cumsum(pos_sorted)
    fp = np.cumsum(~pos_sorted)
    # Only cut the curve where the score changes, so tied samples move together.
    ends = np.r_[np.nonzero(np.diff(score_sorted))[0], len(score_sorted) - 1]
    tpr = np.r_[0.0, tp[ends] / max(tp[-1], 1)]
    fpr = np.r_[0.0, fp[ends] / max(fp[-1], 1)]
    return fpr, tpr


def board_bootstrap_ci(boards, score_fn, n_boot=1000, seed=0):
    """95% interval for each number returned by score_fn(sample_positions) -> {name: value}.

    boards: board id of every sample.  Each round draws as many boards as there are,
    with replacement, and scores all samples of the drawn boards.
    """
    boards = np.asarray(boards)
    names = sorted(set(boards))
    members = {b: np.nonzero(boards == b)[0] for b in names}
    rng = np.random.default_rng(seed)
    draws = {}
    for _ in range(n_boot):
        picked = rng.integers(0, len(names), size=len(names))
        rows = np.concatenate([members[names[i]] for i in picked])
        for key, value in score_fn(rows).items():
            if value is not None and np.isfinite(value):
                draws.setdefault(key, []).append(value)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items() if len(v) >= n_boot // 2}

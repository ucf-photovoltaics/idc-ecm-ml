"""Split boards into a held-out evaluation set and training folds, with no leakage.

What this does, in plain words
------------------------------
All four sensors of one board go to the SAME side. We never split sensors of a
board apart, because they share the same scan, chemistry and test day, so a model
could "cheat" by recognising the board. So we shuffle BOARDS (not rows), set a fifth
of them aside for evaluation, and deal the rest into 5 training folds for
cross-validation (each fold takes a turn as the check set).

It is the same every time: boards are sorted first, then shuffled with a fixed seed.
Boards are dealt out separately for each board type (1, 4, 7, 10), so every type
shows up in every fold.

Run it by hand to check:  uv run python scripts/split.py
"""
import random

SEED = 5937          # fixed seed: the same split on every machine, every run
EVAL_FRACTION = 0.2  # about a fifth of the boards are kept for the final evaluation
N_FOLDS = 5          # training folds for cross-validation


def assign_boards(board_types, seed=SEED, eval_fraction=EVAL_FRACTION, n_folds=N_FOLDS):
    """Give every board a role.

    board_types: dict like {"03_01_0031": 1, ...} (board id -> board type).
    Returns {board_id: -1} for evaluation boards, {board_id: 0..n_folds-1} for training.
    """
    roles = {}
    for btype in sorted(set(board_types.values())):
        boards = sorted(b for b, t in board_types.items() if t == btype)  # sorted: fixed order
        random.Random(f"{seed}-{btype}").shuffle(boards)                  # seeded shuffle
        n_eval = round(eval_fraction * len(boards))
        for b in boards[:n_eval]:
            roles[b] = -1                                                 # -1 = evaluation
        for i, b in enumerate(boards[n_eval:]):
            roles[b] = i % n_folds                                        # deal into folds
    return roles


def check_no_leakage(samples, roles, n_folds=N_FOLDS):
    """Stop with a clear message if any board sits on two sides, or a role is missing."""
    side = {}
    for s in samples:
        role = roles[s["board_id"]]
        if side.setdefault(s["board_id"], role) != role:
            raise SystemExit(f"STOP: board {s['board_id']} is on two sides")
    eval_boards = {b for b, r in side.items() if r == -1}
    train_boards = {b for b, r in side.items() if r >= 0}
    if eval_boards & train_boards:
        raise SystemExit("STOP: a board is in both evaluation and training")
    for f in range(n_folds):
        if not any(r == f for r in side.values()):
            raise SystemExit(f"STOP: fold {f} is empty")
    return eval_boards, train_boards


if __name__ == "__main__":
    import duckdb
    from samples import load_samples
    con = duckdb.connect("build/idc.duckdb", read_only=True)
    samples = load_samples(con)
    roles = assign_boards({s["board_id"]: s["board_type"] for s in samples})
    eval_boards, train_boards = check_no_leakage(samples, roles)
    print(f"{len(eval_boards)} evaluation boards, {len(train_boards)} training boards, no board on both sides")
    print("samples: evaluation", sum(roles[s["board_id"]] == -1 for s in samples),
          "| training", sum(roles[s["board_id"]] >= 0 for s in samples))
    for f in range(N_FOLDS):
        print(f"  fold {f}: {sum(roles[s['board_id']] == f for s in samples)} samples")
    again = assign_boards({s["board_id"]: s["board_type"] for s in samples})
    print("same split when repeated:", again == roles)

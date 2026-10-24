"""List the training samples: one row per sensor test, with the images it needs.

What this does, in plain words
------------------------------
A "sample" is one sensor (U1-U4) on one board that has a measured time-to-failure.
For each sample we also look up:
  * the NEWEST exposed scan of that board (latest date; if two share a date, the
    higher scan number), and
  * the pristine (unused) template for that board's type.
Rows come out in a fixed order, so every build gives the same list.

Run it by hand to check:  uv run python scripts/samples.py
"""
import os
import sys

# Which tests count: the sensor failed or was measured/scanned afterwards.
# 'In progress' (still running) and 'Not started' are left out.
GOOD_STATUS = ("Failed", "Measure/Scanned")

SAMPLES_SQL = """
SELECT t.board_id, t.sensor, t.run_number, t.status, t.ttf_ms,
       b.board_type, ex.repo_path AS exposed_path, pr.repo_path AS pristine_path
FROM tests AS t
JOIN boards AS b ON b.board_id = t.board_id
JOIN (SELECT board_id, repo_path,
             ROW_NUMBER() OVER (PARTITION BY board_id
                                ORDER BY scan_date DESC NULLS LAST, scan_index DESC) AS newest
      FROM images WHERE scan_index >= 1) AS ex
  ON ex.board_id = t.board_id AND ex.newest = 1
JOIN (SELECT board_type, repo_path FROM images WHERE scan_index = 0) AS pr
  ON pr.board_type = b.board_type
WHERE t.status IN ('Failed', 'Measure/Scanned') AND t.ttf_ms IS NOT NULL
ORDER BY t.board_id, t.sensor, t.run_number
"""


def load_samples(con):
    """Return the samples as a list of dicts (one per sensor test), in a fixed order."""
    cur = con.execute(SAMPLES_SQL)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def check_samples(samples, repo_root="."):
    """Stop with a clear message if the list looks wrong."""
    seen = set()
    for s in samples:
        key = (s["board_id"], s["sensor"], s["run_number"])
        if key in seen:
            sys.exit(f"STOP: sample {key} appears twice (is there more than one pristine per board type?)")
        seen.add(key)
        for col in ("exposed_path", "pristine_path"):
            if not os.path.exists(os.path.join(repo_root, s[col])):
                sys.exit(f"STOP: image file missing: {s[col]}")


if __name__ == "__main__":
    import duckdb
    con = duckdb.connect("build/idc.duckdb", read_only=True)
    samples = load_samples(con)
    check_samples(samples)
    boards = sorted({s["board_id"] for s in samples})
    print(f"{len(samples)} samples (sensor tests) from {len(boards)} boards; all image files found")
    for status in GOOD_STATUS:
        print(f"  {status}: {sum(s['status'] == status for s in samples)}")
    for t in (1, 4, 7, 10):
        print(f"  board type {t}: {sum(s['board_type'] == t for s in samples)} samples")
    print("first sample:", samples[0])

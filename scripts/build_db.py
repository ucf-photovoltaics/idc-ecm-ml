"""Build the database from scratch.

Steps: start a fresh database, create the tables from schema.sql, then fill
boards, sensors, electrolytes, measurement_files and tests.
Running it twice gives the same result, because it always starts fresh and
everything is sorted.
"""

import os

import duckdb

from masterlist import read_masterlist
from scan_files import scan_measurement_files

# Folders, found relative to this script so it works from anywhere.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO, "data", "raw")
SCHEMA = os.path.join(REPO, "schema.sql")
MASTERLIST = os.path.join(RAW_DIR, "idc_submersion_masterlist_20250505.csv")
DB_PATH = os.path.join(REPO, "build", "idc.duckdb")


def main():
    # Start fresh: remove any old database file.
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = duckdb.connect(DB_PATH)

    # Create the empty tables from the schema file.
    con.execute(open(SCHEMA).read())

    # Read the measurement files and the masterlist.
    files = scan_measurement_files(RAW_DIR)
    electrolytes, tests = read_masterlist(MASTERLIST)

    # boards: every board ID seen in the files or the masterlist.
    # The board type is the middle number of the ID.
    board_ids = sorted({f["board_id"] for f in files} | {t[1] for t in tests})
    boards = [(b, int(b.split("_")[1])) for b in board_ids]
    con.executemany("INSERT INTO boards VALUES (?, ?)", boards)

    # sensors: every (board, sensor) pair seen in the files or the masterlist.
    pairs = sorted({(f["board_id"], f["sensor"]) for f in files}
                   | {(t[1], t[2]) for t in tests})
    con.executemany("INSERT INTO sensors VALUES (?, ?)", pairs)

    # electrolytes: one per different spelling in the masterlist.
    con.executemany("INSERT INTO electrolytes VALUES (?, ?, ?, ?)", electrolytes)

    # measurement_files: one per CSV. file_id counts 1, 2, 3... in sorted order.
    rows = [
        (i, f["board_id"], f["sensor"], f["kind"], f["file_date"],
         f["sweep_index"], f["repo_path"], f["sha256"])
        for i, f in enumerate(files, start=1)
    ]
    con.executemany("INSERT INTO measurement_files VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    # tests: one per masterlist row.
    con.executemany("INSERT INTO tests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tests)

    # Print how many rows each table now has.
    for table in ["boards", "sensors", "electrolytes", "measurement_files", "tests"]:
        count = con.execute("SELECT count(*) FROM " + table).fetchone()[0]
        print(table, count)
    con.close()


if __name__ == "__main__":
    main()

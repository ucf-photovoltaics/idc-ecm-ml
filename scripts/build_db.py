"""Build the database from scratch.

Steps: start a fresh database, create the tables from schema.sql, then fill
boards, sensors and measurement_files from the files in data/raw.
Running it twice gives the same result, because it always starts fresh and
everything is sorted.
"""

import os

import duckdb

from scan_files import scan_measurement_files

# Folders, found relative to this script so it works from anywhere.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO, "data", "raw")
SCHEMA = os.path.join(REPO, "schema.sql")
DB_PATH = os.path.join(REPO, "build", "idc.duckdb")


def main():
    # Start fresh: remove any old database file.
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = duckdb.connect(DB_PATH)

    # Create the empty tables from the schema file.
    con.execute(open(SCHEMA).read())

    # Read every measurement file in the raw folders.
    files = scan_measurement_files(RAW_DIR)

    # boards: one per board ID. The board type is the middle number of the ID.
    board_ids = sorted({f["board_id"] for f in files})
    boards = [(b, int(b.split("_")[1])) for b in board_ids]
    con.executemany("INSERT INTO boards VALUES (?, ?)", boards)

    # sensors: one per (board, sensor) pair seen in the files.
    pairs = sorted({(f["board_id"], f["sensor"]) for f in files})
    con.executemany("INSERT INTO sensors VALUES (?, ?)", pairs)

    # measurement_files: one per CSV. file_id counts 1, 2, 3... in sorted order.
    rows = [
        (i, f["board_id"], f["sensor"], f["kind"], f["file_date"],
         f["sweep_index"], f["repo_path"], f["sha256"])
        for i, f in enumerate(files, start=1)
    ]
    con.executemany("INSERT INTO measurement_files VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    # Print how many rows each table now has.
    for table in ["boards", "sensors", "measurement_files"]:
        count = con.execute("SELECT count(*) FROM " + table).fetchone()[0]
        print(table, count)
    con.close()


if __name__ == "__main__":
    main()

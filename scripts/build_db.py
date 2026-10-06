"""Build the database from scratch.

Steps: start a fresh database, create the tables from schema.sql, then fill
boards, sensors, electrolytes, measurement_files, tests, the curves and images.
Running it twice gives the same result, because it always starts fresh and
everything is sorted. At the end it writes two files into results/ from the
view v_test_merged: a Parquet file and a CSV with everything (each curve is a
list in one cell), and a short CSV without curves that opens cleanly in Excel.
"""

import os

import duckdb

from curves import read_curve
from images import scan_images
from masterlist import read_masterlist
from parse_filename import parse_filename
from scan_files import scan_measurement_files

# Folders, found relative to this script so it works from anywhere.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO, "data", "raw")
SCHEMA = os.path.join(REPO, "schema.sql")
MASTERLIST = os.path.join(RAW_DIR, "idc_submersion_masterlist_20250505.csv")
DB_PATH = os.path.join(REPO, "build", "idc.duckdb")
RESULTS_DIR = os.path.join(REPO, "results")

# Columns of v_test_merged that hold a single value (no curves). These go into
# the short CSV, because Excel cannot show the long curve lists. The curves
# themselves are given as the path of the raw CSV file they came from.
SUMMARY_COLUMNS = ("board_sensor, board_type, run_number, voltage_v, acid, "
                   "concentration_mm, ph, ttf_ms, final_current_ma, "
                   "exposed_image_path, pristine_image_path, current_file_path, "
                   "cv_exposed_file_path, cv_pristine_file_path, "
                   "cf_exposed_file_path, cf_pristine_file_path")


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
    electrolytes, tests, current_files = read_masterlist(MASTERLIST)
    images = scan_images(RAW_DIR)

    # boards: every board ID seen in the files, the masterlist or the images.
    # The board type is the middle number of the ID.
    board_ids = sorted({f["board_id"] for f in files} | {t[1] for t in tests}
                       | {i["board_id"] for i in images})
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

    # Curves: read each file's numbers and store each CSV column as one list.
    # Which table a file goes into depends on its kind.
    # We also remember the LAST current reading of every current file, and the
    # newest file's last reading for each sensor (used when a date is wrong).
    last_current = {}
    newest_current = {}
    for (file_id, *_), f in zip(rows, files):
        curve = read_curve(os.path.join(REPO, f["repo_path"]), f["kind"])
        if f["kind"] == "CF":
            con.execute("INSERT INTO cf_curves VALUES (?, ?, ?, ?, ?)",
                        [file_id, curve["frequency_hz"], curve["capacitance_raw"],
                         curve["impedance_ohm"], curve["phase_deg"]])
        elif f["kind"] == "CV":
            con.execute("INSERT INTO cv_curves VALUES (?, ?, ?, ?, ?)",
                        [file_id, curve["voltage_v"], curve["capacitance_raw"],
                         curve["impedance_ohm"], curve["phase_deg"]])
        else:
            con.execute("INSERT INTO current_curves VALUES (?, ?, ?)",
                        [file_id, curve["time_ms"], curve["current_ma"]])
            key = (f["board_id"], f["sensor"], f["file_date"])
            last_current[key] = curve["current_ma"][-1]
            sensor_key = (f["board_id"], f["sensor"])
            if sensor_key not in newest_current or f["file_date"] > newest_current[sensor_key][0]:
                newest_current[sensor_key] = (f["file_date"], curve["current_ma"][-1])

    # tests: one per masterlist row. For rows where the masterlist names a
    # current file, final_current_ma is that file's last reading. If the date
    # in the name is wrong, we use the newest current file of the same sensor.
    exact, by_sensor, missed = 0, 0, []
    final_tests = []
    for t in tests:
        test_id, board, sensor = t[0], t[1], t[2]
        if test_id in current_files:
            name = current_files[test_id]
            # Use the date written in the name if the name can be read.
            # Upper-case it so ".CSV" and "_i" spellings work.
            try:
                name_date = parse_filename(name.upper())["file_date"]
            except ValueError:
                name_date = None
            value = last_current.get((board, sensor, name_date))
            if value is not None:
                exact += 1
            elif (board, sensor) in newest_current:
                # Wrong date or odd name: use the sensor's newest current file.
                value = newest_current[(board, sensor)][1]
                by_sensor += 1
            else:
                missed.append(name)
            if value is not None:
                t = t[:6] + (value,) + t[7:]
        final_tests.append(t)
    con.executemany("INSERT INTO tests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", final_tests)
    print("current files named in the masterlist:", len(current_files),
          "| exact match:", exact, "| matched by sensor (date differed):", by_sensor,
          "| no file:", len(missed))
    for name in missed:
        print("  no file:", name)

    # images: one per picture. image_id counts 1, 2, 3... in sorted order.
    image_rows = [
        (n, i["board_id"], i["board_type"], i["scan_date"], i["scan_index"],
         i["repo_path"], i["sha256"])
        for n, i in enumerate(images, start=1)
    ]
    con.executemany("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?)", image_rows)

    # Print how many rows each table now has.
    for table in ["boards", "sensors", "electrolytes", "measurement_files", "tests",
                  "cf_curves", "cv_curves", "current_curves", "images"]:
        count = con.execute("SELECT count(*) FROM " + table).fetchone()[0]
        print(table, count)

    # Write the results files from the merged view, in a fixed order.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    full_path = os.path.join(RESULTS_DIR, "idc_tests_full.parquet")
    full_csv_path = os.path.join(RESULTS_DIR, "idc_tests_full.csv")
    summary_path = os.path.join(RESULTS_DIR, "idc_tests_summary.csv")
    order = "ORDER BY board_sensor, run_number"
    con.execute(f"COPY (SELECT * FROM v_test_merged {order}) TO '{full_path}' (FORMAT PARQUET)")
    con.execute(f"COPY (SELECT * FROM v_test_merged {order}) TO '{full_csv_path}' (HEADER)")
    con.execute(f"COPY (SELECT {SUMMARY_COLUMNS} FROM v_test_merged {order}) "
                f"TO '{summary_path}' (HEADER)")
    n = con.execute("SELECT count(*) FROM v_test_merged").fetchone()[0]
    print("results/idc_tests_full.parquet, idc_tests_full.csv and idc_tests_summary.csv written,",
          n, "rows")
    # Excel keeps at most 32,767 characters per cell. Count the tests whose
    # current curve is longer than that (they look cut off in Excel only).
    too_long = con.execute(
        "SELECT count(*) FROM v_test_merged "
        "WHERE len(CAST(current_time_ms AS VARCHAR)) > 32767 "
        "OR len(CAST(current_ma AS VARCHAR)) > 32767").fetchone()[0]
    print("tests with a curve too long for one Excel cell:", too_long)

    con.close()


if __name__ == "__main__":
    main()

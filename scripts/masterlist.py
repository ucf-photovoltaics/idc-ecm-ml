"""Read the masterlist CSV and clean it up, ready to load into the database.

The masterlist is only used for test conditions and outcomes. File names come
from the folders instead, because the masterlist's file-name columns are not
reliable.
"""

import csv
import re
from datetime import datetime


def to_text(value):
    """Blank cells become None; everything else loses stray spaces."""
    value = value.strip()
    return value if value != "" else None


def to_number(value):
    """Turn text into a decimal number. Blank or non-numeric becomes None."""
    value = to_text(value)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def split_electrolyte(label):
    """Split 'Adipic Acid - 1.24mM' into ('Adipic', 1.24).

    'DI Water' has no acid and no concentration, so it gives (None, None).
    """
    if label.lower().startswith("adipic"):
        acid = "Adipic"
    elif label.lower().startswith("succinic"):
        acid = "Succinic"
    else:
        return None, None
    # The concentration is the number just before "mM".
    found = re.search(r"([\d.]+)\s*mM", label)
    return acid, float(found.group(1))


def read_masterlist(path):
    """Return (electrolytes, tests, current_files).

    electrolytes and tests are lists of tuples in table column order.
    current_files says, for tests whose Current cell holds a file name instead
    of a number, which file that is: {test_id: file name}.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # electrolytes: one row per different spelling, sorted so ids never change.
    labels = sorted({to_text(r["Solution"]) for r in rows} - {None})
    electrolyte_id = {label: i for i, label in enumerate(labels, start=1)}
    electrolytes = [(electrolyte_id[label],) + (label,) + split_electrolyte(label)
                    for label in labels]

    # tests: one row per masterlist row, in file order.
    tests = []
    current_files = {}
    seen = {}  # how many times we have met each (board, sensor)
    for number, r in enumerate(rows, start=1):
        board_id = r["Board ID"].strip()
        sensor = r["Sensor"].strip().upper()
        # A sensor tested twice gets run_number 1, then 2.
        seen[(board_id, sensor)] = seen.get((board_id, sensor), 0) + 1

        ttf = to_number(r["Time to Failure (ms)"])
        # Hand-logged currents are in amps; the database stores mA.
        # Cells holding a file name are not numbers, so they stay blank here.
        current = to_number(r["Current"])
        final_current_ma = round(current * 1000, 6) if current is not None else None
        # A cell that is not a number but not blank holds a current-file name.
        if current is None and to_text(r["Current"]) is not None:
            current_files[number] = to_text(r["Current"])

        date_text = to_text(r["Date"])
        test_date = (datetime.strptime(date_text, "%m/%d/%Y").date()
                     if date_text else None)
        solution = to_text(r["Solution"])

        tests.append((
            number,                                   # test_id
            board_id,
            sensor,
            seen[(board_id, sensor)],                 # run_number
            r["Status"].strip(),
            int(ttf) if ttf is not None else None,    # ttf_ms
            final_current_ma,
            to_text(r["Location"]),
            test_date,
            electrolyte_id[solution] if solution else None,
            to_number(r["Voltage"]),                  # voltage_v
            to_number(r["Ph"]),
            to_text(r["Notes"]),
            number,                                   # source_row
        ))
    return electrolytes, tests, current_files

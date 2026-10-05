"""Read the numbers inside one measurement CSV.

Each CSV column becomes one list. The lists come back named like the columns
in the database tables (cf_curves, cv_curves, current_curves).
"""

import csv

# The header line each kind of file must have, and the database column
# name for each CSV column (in the same order).
LAYOUTS = {
    "CF": (
        ["Frequency (Hz)", "Capacitance (F)", "Impedance (O)", "Phase Angle (D)"],
        ["frequency_hz", "capacitance_raw", "impedance_ohm", "phase_deg"],
    ),
    "CV": (
        ["Voltage (V)", "Capacitance (F)", "Impedance (O)", "Phase Angle (D)"],
        ["voltage_v", "capacitance_raw", "impedance_ohm", "phase_deg"],
    ),
    # Note: in current files the CURRENT comes first and the time second.
    "CURRENT_TIME": (
        ["Current (mA)", "Time (ms)"],
        ["current_ma", "time_ms"],
    ),
}


class EmptyFile(ValueError):
    """The file has a header but no readings (nothing was logged)."""


def read_curve(path, kind):
    """Return {database column name: list of numbers} for one CSV file.

    Some files end with a marker line "#,#,#,#". That line is accepted, but
    only as the very last line, and it is not stored.

    Raises EmptyFile if the file has no readings, and ValueError (naming the
    file) if the header is not the expected one or a row is incomplete or
    not a number.
    """
    expected_header, names = LAYOUTS[kind]
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    if not rows or [h.strip() for h in rows[0]] != expected_header:
        raise ValueError("Unexpected header in " + path)
    readings = rows[1:]
    if not readings:
        raise EmptyFile("No readings in " + path)

    columns = {name: [] for name in names}
    for line_number, row in enumerate(readings, start=2):
        # The end marker "#,#,#,#" is allowed as the last line only.
        if all(cell.strip() == "#" for cell in row) and line_number == len(rows):
            break
        if len(row) != len(names):
            raise ValueError("Line " + str(line_number) + " has the wrong number of values in " + path)
        for name, cell in zip(names, row):
            try:
                columns[name].append(float(cell))
            except ValueError:
                raise ValueError("Line " + str(line_number) + " is not a number in " + path)
    return columns

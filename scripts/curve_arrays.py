"""Pack the current-vs-time and LCR (CV, CF) curves into fixed-size arrays.

What this does, in plain words
------------------------------
Curves have different lengths (a current curve can have 2 points or 6,599), but a
model needs a rectangle of numbers. So every curve is written into a row of the same
width and the unused end of the row is filled with NaN ("not a number" = empty).
Next to each array, `curve_len` says how many points are real.

Five curve sets per sample, in this order (the order of the columns of `curve_len`):
  current      time in ms, current in mA            (whole test, up to the failure)
  cv_exposed   voltage V, capacitance pF, impedance ohm, phase deg   (after exposure)
  cv_pristine  same four columns, before exposure
  cf_exposed   frequency Hz, capacitance pF, impedance ohm, phase deg (after exposure)
  cf_pristine  same four columns, before exposure
A sample with no file for a set has curve_len 0 and a row of NaN.

WARNING: the last time stamp of a `current` curve equals the time to failure
(299 of 305 tests within 5%). Do not use a whole current curve to PREDICT the time
to failure - that gives the answer away. Correlating the curve with other
measurements is fine.

Numbers are stored as float32 (about 7 significant digits).
"""
import numpy as np

COLUMNS = {
    "current": ("current_time_ms", "current_ma"),
    "cv_exposed": ("cv_exposed_voltage_v", "cv_exposed_capacitance", "cv_exposed_impedance", "cv_exposed_phase_deg"),
    "cv_pristine": ("cv_pristine_voltage_v", "cv_pristine_capacitance", "cv_pristine_impedance", "cv_pristine_phase_deg"),
    "cf_exposed": ("cf_exposed_frequency_hz", "cf_exposed_capacitance", "cf_exposed_impedance", "cf_exposed_phase_deg"),
    "cf_pristine": ("cf_pristine_frequency_hz", "cf_pristine_capacitance", "cf_pristine_impedance", "cf_pristine_phase_deg"),
}
SET_NAMES = tuple(COLUMNS)    # fixed order = the columns of curve_len


def load_curves(con):
    """Read every test's curves from the database view v_test_merged.

    Returns {(board_id, sensor, run_number): {set_name: [column lists]}}; a set that
    has no file is left out.
    """
    wanted = [c for cols in COLUMNS.values() for c in cols]
    cur = con.execute("SELECT board_sensor, run_number, " + ", ".join(wanted)
                      + " FROM v_test_merged ORDER BY board_sensor, run_number")
    names = [d[0] for d in cur.description]
    curves = {}
    for row in cur.fetchall():
        r = dict(zip(names, row))
        board_id, sensor = r["board_sensor"].rsplit("_", 1)
        found = {}
        for set_name, cols in COLUMNS.items():
            lists = [r[c] for c in cols]
            if all(x is not None and len(x) > 0 for x in lists):
                found[set_name] = lists
        curves[board_id, sensor, r["run_number"]] = found
    return curves


def max_lengths(curves, samples):
    """Longest curve of each set among the samples (so every file uses the same width)."""
    longest = {name: 1 for name in SET_NAMES}
    for s in samples:
        for name, lists in (curves or {}).get((s["board_id"], s["sensor"], s["run_number"]), {}).items():
            longest[name] = max(longest[name], len(lists[0]))
    return longest


def empty_slots(n, longest):
    """Arrays for n samples, all NaN / zero, ready to be filled in."""
    slots = {name: np.full((n, len(COLUMNS[name]), longest[name]), np.nan, np.float32) for name in SET_NAMES}
    slots["curve_len"] = np.zeros((n, len(SET_NAMES)), np.int64)
    return slots


def fill_slot(slots, j, found):
    """Write one sample's curves (`found`, from load_curves) into row j."""
    for k, name in enumerate(SET_NAMES):
        if name in found:
            columns = np.asarray(found[name], dtype=np.float32)       # shape (columns, points)
            slots[name][j, :, :columns.shape[1]] = columns
            slots["curve_len"][j, k] = columns.shape[1]


def describe_curves():
    """Text for the file notes: which set and column is which."""
    import json
    return json.dumps({"order_of_curve_len": list(SET_NAMES),
                       "columns": {n: [c.split("_", 2)[-1] if n != "current" else c for c in cols]
                                   for n, cols in COLUMNS.items()}})

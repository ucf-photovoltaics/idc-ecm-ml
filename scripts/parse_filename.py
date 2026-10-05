"""Read one measurement filename and pull out its pieces.

Two kinds of names exist in the data:

  03_01_0026_U1_20250117_CF_1.csv   (CV or CF file: 7 pieces)
  03_01_0041_U1_20250418_I.csv      (current-vs-time file: 6 pieces)

Pieces are separated by underscores. Python counts from 0.
"""

from datetime import date


def parse_filename(filename):
    """Return the facts hidden in a filename as a dictionary."""
    # Cut off ".csv" (any capitalisation) so the last piece is clean,
    # then cut the name into pieces at each underscore.
    stem = filename[: -len(".csv")]
    parts = stem.split("_")

    # Refuse names that don't match one of the two known shapes, so an odd
    # name fails loudly instead of being quietly misread.
    is_current = len(parts) == 6 and parts[5] == "I"
    is_curve = len(parts) == 7 and parts[5] in ("CV", "CF")
    if not (is_current or is_curve):
        raise ValueError("Unexpected filename: " + filename)

    # Pieces 0, 1, 2 together are the board ID, e.g. "03_01_0026".
    board_id = "_".join(parts[0:3])
    # Piece 3 is the sensor, e.g. "U1" (a few files write it in lowercase).
    sensor = parts[3].upper()
    # Piece 4 is the date written as YYYYMMDD, e.g. "20250117".
    text = parts[4]
    file_date = date(int(text[0:4]), int(text[4:6]), int(text[6:8]))

    # Piece 5 says what kind of file it is.
    if parts[5] == "I":
        # Current files have no sweep number.
        kind = "CURRENT_TIME"
        sweep_index = None
    else:
        # CV or CF files: piece 5 is the kind, piece 6 is the sweep number.
        kind = parts[5]
        sweep_index = int(parts[6])

    return {
        "board_id": board_id,
        "sensor": sensor,
        "file_date": file_date,
        "kind": kind,
        "sweep_index": sweep_index,
    }

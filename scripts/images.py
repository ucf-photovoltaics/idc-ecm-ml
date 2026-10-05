"""Make a list of every whole-board microscope scan in the raw image folders.

Two kinds of names:
  03_01_0036_20241216_001.jpg   exposed scan: board 03_01_0036, date 2024-12-16, scan 001
  03_01_0246_000.jpg            pristine template: board 03_01_0246, scan 000
Scan 000 is a pristine template (shared by all boards of that board type),
001 or more is an exposed scan. We store the file's path and fingerprint,
not the picture itself.
"""

import os
from datetime import date

from scan_files import fingerprint

# The folders that hold images, relative to the raw data folder.
IMAGE_FOLDERS = [
    "imgscans_exposed",
    "imgscans_pristine",
]


def parse_image_name(filename):
    """Return the facts hidden in an image filename as a dictionary."""
    if not filename.lower().endswith(".jpg"):
        raise ValueError("Unexpected image filename: " + filename)
    parts = filename[: -len(".jpg")].split("_")
    board_id = "_".join(parts[0:3])

    if len(parts) == 5 and len(parts[3]) == 8 and parts[3].isdigit() and parts[4].isdigit():
        # Exposed scan: the 4th piece is the date written as YYYYMMDD.
        text = parts[3]
        scan_date = date(int(text[0:4]), int(text[4:6]), int(text[6:8]))
        scan_index = int(parts[4])
    elif len(parts) == 4 and parts[3].isdigit():
        # Pristine template: no date in the name.
        scan_date = None
        scan_index = int(parts[3])
    else:
        raise ValueError("Unexpected image filename: " + filename)

    return {
        "board_id": board_id,
        "board_type": int(parts[1]),           # middle number of the board ID
        "scan_date": scan_date,
        "scan_index": scan_index,
    }


def scan_images(raw_dir):
    """Return one dictionary per image, sorted by file path."""
    rows = []
    for folder in IMAGE_FOLDERS:
        for name in os.listdir(os.path.join(raw_dir, folder)):
            facts = parse_image_name(name)
            facts["repo_path"] = "data/raw/" + folder + "/" + name
            facts["sha256"] = fingerprint(os.path.join(raw_dir, folder, name))
            rows.append(facts)
    rows.sort(key=lambda row: row["repo_path"])
    return rows

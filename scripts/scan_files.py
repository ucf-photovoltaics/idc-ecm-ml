"""Make a list of every measurement CSV in the raw data folders.

For each file we record what the filename says (board, sensor, date, kind,
sweep), where the file lives, and its fingerprint (sha256).
"""

import hashlib
import os

from parse_filename import parse_filename

# The folders that hold measurement CSVs, relative to the raw data folder.
FOLDERS = [
    "cf/cf_exposed",
    "cf/cf_pristine",
    "cv/cv_exposed",
    "cv/cv_pristine",
    "current_time",
]


def fingerprint(path):
    """Return the sha256 fingerprint of the file at this path."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def scan_measurement_files(raw_dir):
    """Return one dictionary per CSV file, sorted by file path.

    Sorting means every run gives the same list in the same order.
    """
    rows = []
    for folder in FOLDERS:
        for name in os.listdir(os.path.join(raw_dir, folder)):
            facts = parse_filename(name)
            # Where the file lives, written relative to the repo.
            facts["repo_path"] = "data/raw/" + folder + "/" + name
            facts["sha256"] = fingerprint(os.path.join(raw_dir, folder, name))
            rows.append(facts)
    rows.sort(key=lambda row: row["repo_path"])
    return rows

"""Make a list of every microscope image in the raw image folders.

Image names look like  03_01_0036_001_U1.jpg  :
  board 03_01_0036, scan number 001, sensor U1.
Scan 000 is a pristine reference, 001 or more is exposed.
We store the file's path and fingerprint, not the picture itself.
"""

import os

from scan_files import fingerprint

# The folders that hold images, relative to the raw data folder.
IMAGE_FOLDERS = [
    "imgscans_exposed_sensors",
    "imgscans_pristine_sensors",
]


def parse_image_name(filename):
    """Return the facts hidden in an image filename as a dictionary."""
    if not filename.lower().endswith(".jpg"):
        raise ValueError("Unexpected image filename: " + filename)
    parts = filename[: -len(".jpg")].split("_")
    # Expect 5 pieces: board (3 pieces), scan number, sensor.
    if len(parts) != 5 or not parts[3].isdigit():
        raise ValueError("Unexpected image filename: " + filename)
    board_id = "_".join(parts[0:3])
    return {
        "board_id": board_id,
        "board_type": int(parts[1]),           # middle number of the board ID
        "scan_index": int(parts[3]),
        "sensor": parts[4].upper(),
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

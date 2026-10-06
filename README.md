# IDC electrochemical-migration database

Builds a DuckDB database of interdigitated-comb (IDC)
sensor data: time to failure, current vs. time, capacitance/impedance/phase
curves (CV and CF), and before/after microscope images.

## Run

```
just          # builds everything in a container (needs podman)
make all      # same thing on the host (needs uv)
```

The result is `build/idc.duckdb`. It is rebuilt from scratch each time and is
rebuilt only if the scripts, `schema.sql`, or a raw file changed.

Pipeline: `just -> podman -> make -> uv run python scripts/build_db.py -> build/idc.duckdb`

| File | Purpose |
|---|---|
| `Justfile`, `Containerfile`, `Makefile` | run the build in a container |
| `pyproject.toml`, `uv.lock` | Python packages (DuckDB) |
| `schema.sql` | the database schema (copied below) |
| `scripts/build_db.py` | creates the database and loads everything |
| `scripts/parse_filename.py`, `scan_files.py` | read file names, list and fingerprint (sha256) files |
| `scripts/masterlist.py` | clean the masterlist (conditions and time to failure) |
| `scripts/curves.py`, `images.py` | read curve CSVs and image names |
| `results/` | files written by the build (see Results below) |

## Where the raw data lives

`data/raw/` (inputs only, never modified by the build):

- `cf/`, `cv/`: capacitance vs. frequency and vs. voltage, each split into `*_pristine` and `*_exposed`
- `current_time/`: current vs. time logged during each test
- `imgscans_pristine/`, `imgscans_exposed/`: whole-board microscope scans (one pristine template per board type; one or more exposed scans per board; the newest is used)
- `idc_submersion_masterlist_20250505.csv`: the test log (conditions, time to failure)

## What to know about the data

- **File names** carry board, sensor, date, kind and sweep, e.g. `03_01_0026_U1_20250117_CF_1.csv`.
  The masterlist is used only for test conditions and time to failure, because its file-name columns are unreliable.
- **Board ID** `03_01_0041`: the middle number (`01`) is the board type (1, 4, 7, 10). All board types share the same sensor layout. Each board has sensors U1 to U4.
- **Sweep / scan number**: `0` = pristine, `1` or more = exposed (several = additional sweeps). Same for image scan numbers (`000` = pristine).
- **Pristine images** are shared references: one whole-board template per board type.
- **`ttf_ms`** is the time LabVIEW stopped the run. This is the label for the ML task.
- **`final_current_ma`** is the current at that moment. Early runs were logged by hand (in amps, converted to mA); later runs come from the last reading of the LabVIEW current file. If the masterlist date for a file is wrong, the sensor's newest current file is used. A few tests have no current file, and the value is blank.
- **Units**: capacitance is in **pF**, although the file header (written by the LabVIEW program) says F. Impedance is in ohms and phase angle in degrees. The column is called `capacitance_raw` because it holds the numbers exactly as in the file.
- **`v_test_merged`** gives one row per test (tests that were started). When a sensor has several files of one kind, the newest-dated one is used. A sensor tested twice gets the same newest files for both runs. Each test shows its board's newest exposed scan and the pristine template for its board type, so the four sensors of a board share the same image paths.

## Results

Every build also writes three files into `results/`, made from the view `v_test_merged` (one row per started test, 417 rows):

- `idc_tests_full.parquet`: everything, including the full current-vs-time, CV and CF curves (each curve is one list per row). Open it with DuckDB, pandas or Polars.
- `idc_tests_full.csv`: the same table as the Parquet, as a CSV. Each curve is written as a list in one cell, e.g. `[0.0, 0.1, ...]`. Read it with DuckDB, pandas or Polars; see the known issue below before opening it in Excel.
- `idc_tests_summary.csv`: only the one-value columns (board and sensor, voltage, acid, concentration, pH, time to failure, final current) plus the **repo path of each source file**: the current-vs-time CSV, the CV and CF CSVs (exposed and pristine), and the two images. The curves themselves are not in this file; open the file at that path, or use the Parquet. Opens in Excel.


Example, in any DuckDB session: `SELECT board_sensor, ttf_ms FROM 'results/idc_tests_full.parquet' LIMIT 5;`

### Known issue: the curves do not fit in an Excel cell

Excel allows at most 32,767 characters in one cell. A long current-vs-time curve (one test has 6,599 points) is longer than that, so in `idc_tests_full.csv` the end of such a list spills onto the next rows when the file is opened in Excel. The file itself is correct, and DuckDB, pandas and Polars read it fine. The build prints how many tests are affected. `idc_tests_summary.csv` has no curves and opens cleanly in Excel.

## Schema

One row is: a board (`boards`), a sensor on a board (`sensors`), a masterlist
liquid (`electrolytes`), one test of one sensor (`tests`), one CSV file
(`measurement_files`), one CSV's numbers with each column as a list
(`current_curves`, `cv_curves`, `cf_curves`), one whole-board scan (`images`).
Views: `v_test_merged` (one row per test, everything side by side) and
`v_ttf_check` (compares the last logged time with the masterlist time to failure).

```sql
-- boards: one row per physical board.
CREATE TABLE boards (
  board_id    VARCHAR PRIMARY KEY,     -- e.g. '03_01_0041'
  board_type  INTEGER NOT NULL         -- middle number of the ID: 1, 4, 7, 10
);

-- sensors: each board has four sensors, U1 to U4.
CREATE TABLE sensors (
  board_id      VARCHAR NOT NULL REFERENCES boards (board_id),
  sensor        VARCHAR NOT NULL CHECK (sensor IN ('U1','U2','U3','U4')),
  PRIMARY KEY (board_id, sensor)
);

-- electrolytes: the liquid the sensor sat in.
CREATE TABLE electrolytes (
  electrolyte_id    INTEGER PRIMARY KEY,
  raw_label         VARCHAR NOT NULL UNIQUE,  -- exactly as written in the masterlist
  acid              VARCHAR,                  -- 'Adipic' or 'Succinic'; blank for pure DI water
  concentration_mm  DOUBLE                    -- millimolar (mM); blank for pure DI water
);

-- tests: one row per test of one sensor (one masterlist row).
CREATE TABLE tests (
  test_id         INTEGER PRIMARY KEY,
  board_id        VARCHAR NOT NULL,
  sensor          VARCHAR NOT NULL,
  run_number      INTEGER NOT NULL DEFAULT 1,  -- 1 = first test of this sensor
  status          VARCHAR NOT NULL CHECK (status IN
                    ('Not started','In progress','Measure/Scanned','Failed')),
  ttf_ms          INTEGER,                     -- ms; when LabVIEW stopped the run (the ML label)
  final_current_ma DOUBLE,                     -- mA; current at stop (hand-logged amps x1000, or last reading of the current file)
  location        VARCHAR,                     -- lab/room, e.g. 'CREOL-A112'
  test_date       DATE,                        -- from the masterlist's M/D/YYYY
  electrolyte_id  INTEGER REFERENCES electrolytes (electrolyte_id),
  voltage_v       DOUBLE,                      -- applied bias, volts
  ph              DOUBLE,                      -- pH, when recorded
  notes           VARCHAR,                     -- free text, as written
  source_row      INTEGER NOT NULL,            -- row number in the masterlist
  FOREIGN KEY (board_id, sensor) REFERENCES sensors (board_id, sensor),
  UNIQUE (board_id, sensor, run_number),
  CHECK (status <> 'Not started' OR ttf_ms IS NULL)
);

-- measurement_files: one row per CSV found in the folders, read from the filename.
CREATE TABLE measurement_files (
  file_id       INTEGER PRIMARY KEY,
  board_id      VARCHAR NOT NULL,
  sensor        VARCHAR NOT NULL,
  kind          VARCHAR NOT NULL CHECK (kind IN ('CV','CF','CURRENT_TIME')),
  file_date     DATE NOT NULL,                -- date in the filename
  sweep_index   INTEGER CHECK (sweep_index >= 0),  -- 0 = pristine, 1+ = exposed; blank for current files
  repo_path     VARCHAR NOT NULL UNIQUE,      -- where the file lives in the repo
  sha256        VARCHAR NOT NULL,             -- fingerprint of the raw file
  FOREIGN KEY (board_id, sensor) REFERENCES sensors (board_id, sensor),
  UNIQUE (board_id, sensor, kind, file_date, sweep_index)
);

-- current_curves: current vs time for one test file. One row per file, one list per CSV column.
CREATE TABLE current_curves (
  file_id     INTEGER PRIMARY KEY REFERENCES measurement_files (file_id),
  time_ms     DOUBLE[] NOT NULL,              -- ms since the file started
  current_ma  FLOAT[]  NOT NULL,              -- mA
  CHECK (len(time_ms) = len(current_ma))
);

-- cv_curves: capacitance, impedance, phase vs voltage for one file.
CREATE TABLE cv_curves (
  file_id          INTEGER PRIMARY KEY REFERENCES measurement_files (file_id),
  voltage_v        DOUBLE[] NOT NULL,         -- volts
  capacitance_raw  FLOAT[]  NOT NULL,         -- pF (the file header says F, but the values are pF)
  impedance_ohm    FLOAT[]  NOT NULL,         -- ohms
  phase_deg        FLOAT[]  NOT NULL,         -- degrees
  CHECK (len(voltage_v) = len(capacitance_raw)
         AND len(voltage_v) = len(impedance_ohm)
         AND len(voltage_v) = len(phase_deg))
);

-- cf_curves: capacitance, impedance, phase vs frequency for one file.
CREATE TABLE cf_curves (
  file_id          INTEGER PRIMARY KEY REFERENCES measurement_files (file_id),
  frequency_hz     DOUBLE[] NOT NULL,         -- hertz
  capacitance_raw  FLOAT[]  NOT NULL,         -- pF (the file header says F, but the values are pF)
  impedance_ohm    FLOAT[]  NOT NULL,         -- ohms
  phase_deg        FLOAT[]  NOT NULL,         -- degrees
  CHECK (len(frequency_hz) = len(capacitance_raw)
         AND len(frequency_hz) = len(impedance_ohm)
         AND len(frequency_hz) = len(phase_deg))
);

-- images: whole-board microscope scans (the pictures stay as files).
CREATE TABLE images (
  image_id    INTEGER PRIMARY KEY,
  board_id    VARCHAR NOT NULL REFERENCES boards (board_id),
  board_type  INTEGER NOT NULL,             -- copied from the board ID
  scan_date   DATE,                         -- date in the filename; blank for pristine templates
  scan_index  INTEGER NOT NULL CHECK (scan_index >= 0),  -- 0 = pristine template (shared per board type), 1+ = exposed
  repo_path   VARCHAR NOT NULL UNIQUE,
  sha256      VARCHAR NOT NULL,
  UNIQUE (board_id, scan_date, scan_index)
);
```

The full file, including the views, is `schema.sql`.

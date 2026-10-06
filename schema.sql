-- schema_v3.sql: IDC electrochemical-migration database (DuckDB)
-- Assumptions and known facts about the data are listed in the README.

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

-- v_ttf_check: last time in each current file vs the masterlist ttf. Big differences need a look.
CREATE VIEW v_ttf_check AS
SELECT f.file_id, f.board_id, f.sensor, f.repo_path,
       list_max(c.time_ms)             AS last_time_ms,
       t.ttf_ms                        AS masterlist_ttf_ms,
       list_max(c.time_ms) - t.ttf_ms  AS difference_ms
FROM measurement_files f
JOIN current_curves c ON c.file_id = f.file_id
JOIN tests t ON t.board_id = f.board_id AND t.sensor = f.sensor
WHERE f.kind = 'CURRENT_TIME';

-- v_test_merged: ONE ROW PER TEST. Newest-dated file wins per sensor, kind and
-- pristine/exposed. 'Not started' tests are left out.
-- Limitation: a sensor tested twice gets the same newest files for both runs.
CREATE VIEW v_test_merged AS
WITH newest AS (
  SELECT f.*, (f.sweep_index >= 1) AS is_exposed
  FROM measurement_files f
  QUALIFY row_number() OVER (
            PARTITION BY f.board_id, f.sensor, f.kind, (f.sweep_index >= 1)
            ORDER BY f.file_date DESC, f.sweep_index DESC NULLS LAST) = 1
),
newest_exposed_image AS (
  SELECT * FROM images
  WHERE scan_index >= 1
  QUALIFY row_number() OVER (
            PARTITION BY board_id ORDER BY scan_date DESC, scan_index DESC) = 1
)
SELECT
  t.board_id || '_' || t.sensor  AS board_sensor,
  b.board_type,
  t.run_number,
  t.voltage_v,
  e.acid,
  e.concentration_mm,
  t.ph,
  t.ttf_ms,
  t.final_current_ma,
  cur.time_ms    AS current_time_ms,
  cur.current_ma AS current_ma,
  cvx.voltage_v AS cv_exposed_voltage_v,  cvx.capacitance_raw AS cv_exposed_capacitance,
  cvx.impedance_ohm AS cv_exposed_impedance,  cvx.phase_deg AS cv_exposed_phase_deg,
  cvp.voltage_v AS cv_pristine_voltage_v, cvp.capacitance_raw AS cv_pristine_capacitance,
  cvp.impedance_ohm AS cv_pristine_impedance, cvp.phase_deg AS cv_pristine_phase_deg,
  cfx.frequency_hz AS cf_exposed_frequency_hz, cfx.capacitance_raw AS cf_exposed_capacitance,
  cfx.impedance_ohm AS cf_exposed_impedance,   cfx.phase_deg AS cf_exposed_phase_deg,
  cfp.frequency_hz AS cf_pristine_frequency_hz, cfp.capacitance_raw AS cf_pristine_capacitance,
  cfp.impedance_ohm AS cf_pristine_impedance,   cfp.phase_deg AS cf_pristine_phase_deg,
  xi.repo_path AS exposed_image_path,
  pi.repo_path AS pristine_image_path,
  nf_cur.repo_path AS current_file_path,
  nf_cvx.repo_path AS cv_exposed_file_path,
  nf_cvp.repo_path AS cv_pristine_file_path,
  nf_cfx.repo_path AS cf_exposed_file_path,
  nf_cfp.repo_path AS cf_pristine_file_path
FROM tests t
JOIN boards b ON b.board_id = t.board_id
LEFT JOIN electrolytes e ON e.electrolyte_id = t.electrolyte_id
LEFT JOIN newest nf_cur ON nf_cur.board_id = t.board_id AND nf_cur.sensor = t.sensor
                       AND nf_cur.kind = 'CURRENT_TIME'
LEFT JOIN current_curves cur ON cur.file_id = nf_cur.file_id
LEFT JOIN newest nf_cvx ON nf_cvx.board_id = t.board_id AND nf_cvx.sensor = t.sensor
                       AND nf_cvx.kind = 'CV' AND nf_cvx.is_exposed
LEFT JOIN cv_curves cvx ON cvx.file_id = nf_cvx.file_id
LEFT JOIN newest nf_cvp ON nf_cvp.board_id = t.board_id AND nf_cvp.sensor = t.sensor
                       AND nf_cvp.kind = 'CV' AND NOT nf_cvp.is_exposed
LEFT JOIN cv_curves cvp ON cvp.file_id = nf_cvp.file_id
LEFT JOIN newest nf_cfx ON nf_cfx.board_id = t.board_id AND nf_cfx.sensor = t.sensor
                       AND nf_cfx.kind = 'CF' AND nf_cfx.is_exposed
LEFT JOIN cf_curves cfx ON cfx.file_id = nf_cfx.file_id
LEFT JOIN newest nf_cfp ON nf_cfp.board_id = t.board_id AND nf_cfp.sensor = t.sensor
                       AND nf_cfp.kind = 'CF' AND NOT nf_cfp.is_exposed
LEFT JOIN cf_curves cfp ON cfp.file_id = nf_cfp.file_id
LEFT JOIN newest_exposed_image xi ON xi.board_id = t.board_id
LEFT JOIN images pi ON pi.board_type = b.board_type AND pi.scan_index = 0
WHERE t.status <> 'Not started';

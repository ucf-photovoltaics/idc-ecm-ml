.RECIPEPREFIX = >
.PHONY: all check clean

# Every raw input file. If any of them changes, everything is rebuilt.
RAW := $(shell find data/raw -type f)
# The scripts that turn the database into pictures and labels.
TENSOR_SRC := scripts/make_tensors.py scripts/crop_sensors.py scripts/samples.py scripts/split.py
# The scripts that build the database (everything except the tensor and check scripts).
DB_SRC := $(filter-out $(TENSOR_SRC) scripts/check.py,$(wildcard scripts/*.py))

all: build/idc.duckdb data/inputs/train.safetensors data/inputs/eval.safetensors

# The database is rebuilt only if the scripts, the schema or any raw file changed.
build/idc.duckdb: $(DB_SRC) schema.sql $(RAW)
> uv run python scripts/build_db.py

# One run writes both files (the "&:" means "these two come from one recipe").
data/inputs/train.safetensors data/inputs/eval.safetensors &: build/idc.duckdb $(TENSOR_SRC)
> uv run python scripts/make_tensors.py

# Check the files, then prove a second build would do nothing (make -q = "is it up to date?").
check: all
> uv run python scripts/check.py
> @$(MAKE) --question all && echo "idempotent: a second build has nothing to redo"

clean:
> rm -rf build data/inputs

.PHONY: all clean

RAW := $(shell find data/raw -type f)

all: build/idc.duckdb

# The database is rebuilt only if the scripts, the schema or any raw file changed.
build/idc.duckdb: scripts/*.py schema.sql $(RAW)
	uv run python scripts/build_db.py

clean:
	rm -rf build

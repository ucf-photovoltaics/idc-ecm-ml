.RECIPEPREFIX = >
.PHONY: all check clean

# How Python is started. Inside the container this is uv; to try the pipeline on a machine without uv,
# run e.g.  make PY=python3
PY ?= uv run python

# Every raw input file. If any of them changes, everything is rebuilt.
RAW := $(shell find data/raw -type f)

# The scripts that turn the database into pictures and labels.
TENSOR_SRC := scripts/make_tensors.py scripts/crop_sensors.py scripts/samples.py scripts/split.py
# The scripts that train the model, score it, and draw the figures / report.
MODEL_SRC  := scripts/data_io.py scripts/model.py scripts/train.py
SCORE_SRC  := scripts/metrics.py scripts/evaluate.py
FIG_SRC    := scripts/miniplot.py scripts/make_figures.py
CHECK_SRC  := scripts/check.py scripts/check_results.py
# The scripts that build the database (everything that is not in one of the lists above).
DB_SRC := $(filter-out $(TENSOR_SRC) $(MODEL_SRC) $(SCORE_SRC) $(FIG_SRC) scripts/make_report.py $(CHECK_SRC),$(wildcard scripts/*.py))

INPUTS  := data/inputs/train.safetensors data/inputs/eval.safetensors
MODEL   := build/model/model.npz build/model/history.json
FIGURES := results/figures/training_curve.pdf results/figures/roc.pdf results/figures/parity.pdf

# The whole chain, in order:
#   raw files -> database -> tensors -> trained model -> results.json -> figures + numbers -> report.pdf
all: build/idc.duckdb $(INPUTS) results/report.pdf

# The database is rebuilt only if the scripts, the schema or any raw file changed.
build/idc.duckdb: $(DB_SRC) schema.sql $(RAW)
> $(PY) scripts/build_db.py

# One run writes both files (the "&:" means "these come from one recipe").
$(INPUTS) &: build/idc.duckdb $(TENSOR_SRC)
> $(PY) scripts/make_tensors.py

# Training reads ONLY the training file. The held-out file is left out of this list on purpose, so
# the model cannot depend on it. New data -> new train.safetensors -> the model is retrained.
$(MODEL) &: data/inputs/train.safetensors $(MODEL_SRC)
> $(PY) scripts/train.py

# Scoring is the only step that opens the held-out file. Writes every number the report uses.
results/results.json: $(MODEL) $(INPUTS) $(MODEL_SRC) $(SCORE_SRC)
> $(PY) scripts/evaluate.py

# Figures are drawn from results.json alone.
$(FIGURES) &: results/results.json $(FIG_SRC)
> $(PY) scripts/make_figures.py

# The report's numbers are also taken from results.json alone.
build/report/numbers.tex: results/results.json scripts/make_report.py
> $(PY) scripts/make_report.py

# pdflatex is run twice so that figure and table references resolve. The log is kept, and shown if it fails.
results/report.pdf: report/report.tex build/report/numbers.tex $(FIGURES)
> mkdir -p build/report
> pdflatex -interaction=nonstopmode -halt-on-error -output-directory build/report report/report.tex > build/report/pdflatex.log || { tail -n 40 build/report/pdflatex.log; exit 1; }
> pdflatex -interaction=nonstopmode -halt-on-error -output-directory build/report report/report.tex > build/report/pdflatex.log || { tail -n 40 build/report/pdflatex.log; exit 1; }
> cp build/report/report.pdf results/report.pdf
> @echo "wrote results/report.pdf"

# Check the files and the results, then prove a second build would do nothing (make -q = "is it up to date?").
check: all
> $(PY) scripts/check.py
> $(PY) scripts/check_results.py
> @$(MAKE) --question all && echo "idempotent: a second build has nothing to redo"

clean:
> rm -rf build data/inputs results/results.json results/report.pdf results/figures

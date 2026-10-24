# Small Python image that already includes uv (the tool that installs packages).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# make runs the build steps in the Makefile.
RUN apt-get update \
 && apt-get install -y --no-install-recommends make \
 && rm -rf /var/lib/apt/lists/*

# Keep the packages inside the image (not in your folder), install them once now,
# and never go online when the pipeline runs.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_FROZEN=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /work
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

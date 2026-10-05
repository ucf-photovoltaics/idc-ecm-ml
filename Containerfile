# Small Python image that already includes uv (the tool that installs packages).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# make runs the build steps in the Makefile.
RUN apt-get update \
 && apt-get install -y --no-install-recommends make \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

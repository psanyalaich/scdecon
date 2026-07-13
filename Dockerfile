# syntax=docker/dockerfile:1
#
# Single-stage image for running scdecon (CLI + Snakemake pipeline) reproducibly.
# The package is installed from source with the "pipeline" extra so both the
# `scdecon` CLI and the Snakemake workflow work inside the container.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the package first (from the metadata + source only) for better layer
# caching: this layer is rebuilt only when the package or its dependencies
# change, not when the workflow/scripts change. The dynamic version is read from
# src/scdecon/__init__.py, and the readme/license are referenced by pyproject.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[pipeline]"

# Orchestration and dataset-specific code (not part of the installable package,
# but needed to run the Snakemake workflow and the melanoma workflow in-container).
COPY workflow ./workflow
COPY config ./config
COPY scripts ./scripts

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 scdecon \
    && chown -R scdecon:scdecon /app
USER scdecon

# Default to the CLI help. Override the whole command to do something else, e.g.:
#   docker run --rm IMAGE scdecon version
#   docker run --rm -v "$PWD:/data" IMAGE scdecon deconvolve --config /data/run.yaml
#   docker run --rm IMAGE snakemake --cores 1 --config run_config=config/example_run.yaml
CMD ["scdecon", "--help"]

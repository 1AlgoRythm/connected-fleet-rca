#!/usr/bin/env bash
# End-to-end: simulate -> MQTT -> SQLite -> analytics -> README tables. ~10 minutes on an M-series Mac.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync
uv run python -m fleet.run_transport                          # results/ingest_report.json, ground_truth.json
uv run python -m fleet.run_transport --default-puback-timeout  # negative control
uv run python -m fleet.run_analytics                          # metrics, CV, RCA, leakage checks
uv run python scripts/make_readme_tables.py

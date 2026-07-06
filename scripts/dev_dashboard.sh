#!/usr/bin/env bash
# Dev dashboard with auto-reload.
#
# Restarts `dfc-dashboard` whenever a .py under src/ changes. A full PROCESS restart
# (not a Streamlit "rerun on save") is used deliberately: the dashboard's logic lives
# in imported modules held by @st.cache_resource singletons (the Rerun servers, the
# run registry) and background emitter threads, none of which hot-swap on a Streamlit
# rerun. Restarting the process is the only reliable way to pick up changes to those.
#
# Usage:   scripts/dev_dashboard.sh [extra dfc-dashboard args]
# Stop:    Ctrl-C (or kill the watchmedo process)
set -euo pipefail

# Conda env that has watchdog (watchmedo) + this package installed (-e).
ENV_BIN="${DFC_ENV_BIN:-/home/flexiv/miniforge3/envs/dual-flexiv-control/bin}"
cd "$(dirname "$0")/.."

echo "[dev] watching src/dual_flexiv_control for *.py changes -> restart dfc-dashboard"
exec "$ENV_BIN/watchmedo" auto-restart \
  --directory=./src/dual_flexiv_control \
  --pattern='*.py' \
  --ignore-patterns='*/__pycache__/*;*.pyc' \
  --recursive \
  --debounce-interval=1 \
  --signal=SIGTERM \
  --kill-after=4 \
  -- "$ENV_BIN/dfc-dashboard" --server.address=0.0.0.0 "$@"

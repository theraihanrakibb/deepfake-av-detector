#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TZ=Asia/Shanghai
logs="logs/$(date +%F)/pytest/run_$(date +%H-%M-%S)"
mkdir -p "$logs"
.venv/bin/python -m pytest -q tests "$@" 2>&1 | tee "$logs/output.log"

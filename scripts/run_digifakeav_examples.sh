#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
training_run=${1:?Pass the WildDeepfake run results directory}
export TZ=Asia/Shanghai OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
day=$(date +%F)
run=run_$(date +%H-%M-%S)
logs="logs/$day/digifakeav_examples/$run"
results="results/$day/digifakeav_examples/$run"
mkdir -p "$logs" "$results"
exec >"$logs/output.log" 2>&1
printf 'Waiting for the WildDeepfake run at %s\n' "$training_run"
deadline=$((SECONDS + 1800))
until [[ -f "$training_run/complete" ]]; do
    if (( SECONDS >= deadline )); then
        echo 'Timed out waiting for the WildDeepfake run.'
        exit 1
    fi
    sleep 10
done
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
.venv/bin/python -u -m scripts.evaluate_saved_checkpoints "$results" \
    --weights-dir "$training_run/results/weights" \
    --data-dir data/digifakeav_official_examples --require-audio
touch "$results/complete"

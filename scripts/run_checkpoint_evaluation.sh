#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TZ=Asia/Shanghai
day=$(date +%F)
run=run_$(date +%H-%M-%S)
logs="logs/$day/checkpoint_smoke/$run"
results="results/$day/checkpoint_smoke/$run"
mkdir -p "$logs" "$results"
exec >"$logs/output.log" 2>&1
echo 'Waiting for the four checkpoints and 260 video clips.'
checkpoints_ready() {
    [[ $(stat -c %s weights/trained_model_lipsync_best.pth 2>/dev/null) == 547212117 &&
       $(stat -c %s weights/trained_model_mesonet_best.pth 2>/dev/null) == 142997 &&
       $(stat -c %s weights/trained_model_multimodal_best.pth 2>/dev/null) == 486662203 &&
       $(stat -c %s weights/trained_model_xceptionnet_best.pth 2>/dev/null) == 83558701 ]]
}
deadline=$((SECONDS + 7200))
until checkpoints_ready && [[ $(find data -type f -name '*.mp4' | wc -l) == 260 ]]; do
    if (( SECONDS >= deadline )); then
        echo 'Timed out waiting for the project transfer.'
        exit 1
    fi
    sleep 20
done
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2,2nr | cut -d, -f1 | head -n 1)
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
.venv/bin/python -u -m scripts.evaluate_saved_checkpoints "$results"
touch "$results/complete"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ROOT="${ROOT:-results/2026-11-22/dfa}"
DFA="data/digifakeav_subset"
DFA_M="$DFA/dfa.json"
DFA_S="$DFA/dfa_split.json"
SEEDS="${SEEDS:-0 1 2 3}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
GPUS=(0 1 2 3)
mkdir -p "$ROOT/locks" logs

export ROOT DFA DFA_M DFA_S SEEDS EPOCHS BATCH_SIZE MIN_FREE_MIB
export HF_ENDPOINT=https://hf-mirror.com MPLBACKEND=Agg
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PRETRAINED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1

command -v flock >/dev/null || { echo "flock missing"; exit 1; }
.venv/bin/python -c "import os, subprocess; rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines(); free={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows}; min_free=int(os.environ['MIN_FREE_MIB']); assert all(free.get(i,0)>=min_free for i in range(4)), free; print('GPU free MiB:', free)"

cp -n "$DFA/manifest.json" "$DFA_M"
cp -n "$DFA/video_split.json" "$DFA_S"
cmp -s "$DFA/manifest.json" "$DFA_M" || { echo "dfa.json differs from legacy manifest"; exit 1; }
cmp -s "$DFA/video_split.json" "$DFA_S" || { echo "dfa_split.json differs from legacy split"; exit 1; }
.venv/bin/python scripts/validate_dataset_splits.py "$DFA_M"

for gpu in "${GPUS[@]}"; do : > "$ROOT/locks/gpu$gpu.lock"; done
: > "$ROOT/train_jobs.tsv"
i=0
for seed in $SEEDS; do
  gpu=${GPUS[$((i % 4))]}
  printf 'dfa %s %s\n' "$gpu" "$seed" >> "$ROOT/train_jobs.tsv"
  i=$((i + 1))
done

xargs -P 4 -n 3 bash -c '
set -euo pipefail
ds=$1; gpu=$2; seed=$3
data="$DFA"; manifest="$DFA_M"; split="$DFA_S"; require=1
name="${ds}_s${seed}_eval.json"
flock "$ROOT/locks/gpu${gpu}.lock" env \
  CUDA_VISIBLE_DEVICES="$gpu" GPU_ID="$gpu" DATA="$data" MANIFEST="$manifest" SPLIT="$split" \
  RUN_ROOT="$ROOT/$ds" REQUIRE_AUDIO="$require" SEEDS="$seed" EVALUATION_NAME="$name" \
  EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" MIN_FREE_MIB="$MIN_FREE_MIB" \
  bash scripts/run_dataset_worker.sh > "$ROOT/${ds}_s${seed}_g${gpu}.log" 2>&1
' _ < "$ROOT/train_jobs.tsv"

for seed in $SEEDS; do
  test -f "$ROOT/dfa/seed_$seed/evaluation/dfa_s${seed}_eval.json"
done

if [ "${WRITE_DONE:-1}" -eq 1 ]; then
  printf '{"status":"done","dataset":["dfa"],"gpus":[0,1,2,3],"seeds":"%s","threshold":0.5,"audio":"dfa real"}\n' "$SEEDS" > "$ROOT/done.json"
fi
echo "DFA-only done: $ROOT"

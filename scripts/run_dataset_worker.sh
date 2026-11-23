#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
DATA="${DATA:-data/digifakeav_subset}"
MANIFEST="${MANIFEST:-$DATA/dfa.json}"
SPLIT="${SPLIT:-$DATA/dfa_split.json}"
RUN_ROOT="${RUN_ROOT:-results/2026-11-22/all4/dfa}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEEDS="${SEEDS:-0 1 2}"
REQUIRE_AUDIO="${REQUIRE_AUDIO:-1}"
EVALUATION_NAME="${EVALUATION_NAME:-dfa_eval.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
NO_PRETRAINED="${NO_PRETRAINED:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
RUN_CONTROLS="${RUN_CONTROLS:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
CLASS_WEIGHTED_LOSS="${CLASS_WEIGHTED_LOSS:-0}"
mkdir -p "$RUN_ROOT" logs

# shortcut: failed partial outputs remain; clean after verified results.

FREE_MIB=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F',' -v id="$GPU_ID" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1 == id) print $2}')
if [ -z "$FREE_MIB" ] || [ "$FREE_MIB" -lt "$MIN_FREE_MIB" ]; then
  echo "GPU $GPU_ID has insufficient free memory: ${FREE_MIB:-unknown} MiB"
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"

AUDIO_ARGS=()
if [ "$REQUIRE_AUDIO" -eq 1 ]; then
  AUDIO_ARGS=(--require-audio)
fi
PRETRAINED_ARGS=(--pretrained)
if [ "$NO_PRETRAINED" -eq 1 ]; then
  PRETRAINED_ARGS=(--no-pretrained)
fi

test -f "$MANIFEST"
test -f "$SPLIT"
.venv/bin/python scripts/validate_dataset_splits.py "$MANIFEST"

TRAIN_EXTRA_ARGS=()
if [ "$CLASS_WEIGHTED_LOSS" -eq 1 ]; then
  TRAIN_EXTRA_ARGS=(--class-weighted-loss)
fi

for seed in $SEEDS; do
  OUT="$RUN_ROOT/seed_$seed"
  mkdir -p "$OUT"

  if [ "$RUN_TRAIN" -eq 1 ]; then
    echo "Training all models with seed $seed"
    PYTHONPATH=src MPLBACKEND=Agg \
      .venv/bin/python src/train_detectors.py \
      --mode all \
      --split-manifest "$SPLIT" \
      --output-dir "$OUT" \
      "${PRETRAINED_ARGS[@]}" \
      "${AUDIO_ARGS[@]}" \
      "${TRAIN_EXTRA_ARGS[@]}" \
      --skip-face-crop \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --seed "$seed"
  fi

  if [ "$RUN_ABLATIONS" -eq 1 ]; then
    echo "Running ablations with seed $seed"
    PYTHONPATH=src MPLBACKEND=Agg \
      .venv/bin/python src/run_ablations.py \
      --split-manifest "$SPLIT" \
      --output-dir "$OUT/ablations" \
      "${PRETRAINED_ARGS[@]}" \
      "${AUDIO_ARGS[@]}" \
      --skip-face-crop \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --seed "$seed"
  fi

  if [ "$RUN_CONTROLS" -eq 1 ]; then
    echo "Running controls with seed $seed"
    PYTHONPATH=src MPLBACKEND=Agg \
      .venv/bin/python src/evaluate_controls.py \
      --split-manifest "$MANIFEST" \
      --checkpoint "$OUT/weights/trained_model_multimodal_best.pth" \
      --output "$OUT/${EVALUATION_NAME%.json}_ctrl.json" \
      "${AUDIO_ARGS[@]}" \
      --skip-face-crop \
      --batch-size "$BATCH_SIZE"
  fi

  if [ "$RUN_EVAL" -eq 1 ]; then
    echo "Evaluating saved checkpoints with seed $seed"
    PYTHONPATH=src MPLBACKEND=Agg \
      .venv/bin/python scripts/evaluate_saved_checkpoints.py \
      "$OUT/evaluation" \
      --weights-dir "$OUT/weights" \
      --split-manifest "$SPLIT" \
      --output-name "$EVALUATION_NAME" \
      "${AUDIO_ARGS[@]}" \
      --skip-face-crop
    test -f "$OUT/evaluation/$EVALUATION_NAME"
  fi
done

echo "Done: $DATA seed(s) $SEEDS GPU $GPU_ID"

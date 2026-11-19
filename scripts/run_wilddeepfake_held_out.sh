#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
project=$PWD
export TZ=Asia/Shanghai OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$project"
day=$(date +%F)
run=run_$(date +%H-%M-%S)
logs="$project/logs/$day/wilddeepfake_held_out/$run"
results="$project/results/$day/wilddeepfake_held_out/$run"
mkdir -p "$logs" "$results"
exec >"$logs/output.log" 2>&1
cd "$results"
python="$project/.venv/bin/python"
data="$project/data/wilddeepfake"
[[ $(find "$data/real" -type f -name '*.mp4' | wc -l) == 52 ]]
[[ $(find "$data/fake" -type f -name '*.mp4' | wc -l) == 58 ]]
find "$data" -name '*.mp4' -type f -print0 | sort -z | xargs -0 sha256sum > input_sha256.txt
[[ -z $(cut -d' ' -f1 input_sha256.txt | sort | uniq -d) ]]
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
"$python" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.__version__)'
"$python" -u "$project/src/train_detectors.py" --mode all --data_dir "$data" \
    --epochs 10 --patience 3 --batch_size 2 --seed 42
"$python" -u -m scripts.evaluate_saved_checkpoints "$results" \
    --weights-dir results/weights --split-manifest results/docs/video_split.json
touch complete

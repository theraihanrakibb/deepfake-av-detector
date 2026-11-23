import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_transfer_flags_are_available():
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_saved_checkpoints.py"), "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--silent-audio" in result.stdout
    assert "--output-name" in result.stdout


def test_dataset_launchers_are_separate():
    dfa = (ROOT / "scripts" / "run_digifakeav.sh").read_text()
    wdf = (ROOT / "scripts" / "run_wilddeepfake.sh").read_text()
    for script in (dfa, wdf):
        assert "GPUS=(0 1 2 3)" in script
        assert 'SEEDS="${SEEDS:-0 1 2 3}"' in script
        assert "-P 4" in script
        assert "scripts/validate_dataset_splits.py" in script
        assert "scripts/run_dataset_worker.sh" in script
        assert "/home/hyang" not in script
    assert 'ROOT="${ROOT:-results/2026-11-22/dfa}"' in dfa
    assert 'ROOT="${ROOT:-results/2026-11-22/wdf}"' in wdf
    assert "data/digifakeav_subset" in dfa
    assert "data/wilddeepfake" in wdf
    assert "wdf110_split.json" in wdf


def test_thesis_figure_generator_is_available():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_thesis_figures.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "figure generator check: ok" in result.stdout


def test_improved_run_controls_and_threshold_tuning():
    worker = (ROOT / "scripts" / "run_dataset_worker.sh").read_text()
    assert "RUN_ABLATIONS" in worker
    assert "CLASS_WEIGHTED_LOSS" in worker
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in worker
    assert "/home/hyang" not in worker
    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate_saved_checkpoints import pick_threshold

    threshold = pick_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.2 < threshold < 0.8



def test_training_stability_defaults():
    worker = (ROOT / "scripts" / "run_dataset_worker.sh").read_text()
    assert 'NO_PRETRAINED="${NO_PRETRAINED:-0}"' in worker
    for script in (ROOT / "scripts" / "run_digifakeav.sh", ROOT / "scripts" / "run_wilddeepfake.sh"):
        assert "NO_PRETRAINED=0" in script.read_text()
    train = (ROOT / "src" / "train_detectors.py").read_text()
    assert train.count("clip_grad_norm_") >= 3
    assert "weight_decay=1e-4" in train
    assert "has_both_classes = len(set(int(label) for label in train_labels)) == 2" in train
    datasets = (ROOT / "src" / "data_loader" / "datasets.py").read_text()
    assert "TF.rotate" not in datasets
    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate_saved_checkpoints import calibrated_ensemble, MODEL_NAMES

    metrics = {name: {"auc": 0.9 if name == "multimodal" else 0.5} for name in MODEL_NAMES}
    weights = calibrated_ensemble(metrics)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert min(weights.values()) > 0.0
    assert weights["multimodal"] < 0.75

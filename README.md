# Detecting Deepfake Videos by a Multimodal Audio-Visual Framework with Temporal Inconsistencies

<p align="center">
  <img src="https://img.shields.io/badge/Thesis-NWPU-blue" alt="NWPU Thesis"/>
  <img src="https://img.shields.io/github/license/theraihanrakibb/deepfake-av-detector" alt="License"/>
  <img src="https://img.shields.io/github/stars/theraihanrakibb/deepfake-av-detector" alt="Stars"/>
  <img src="https://img.shields.io/github/forks/theraihanrakibb/deepfake-av-detector" alt="Forks"/>
  <img src="https://img.shields.io/github/last-commit/theraihanrakibb/deepfake-av-detector" alt="Last Commit"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Dataset-WildDeepfake-orange" alt="WildDeepfake"/>
</p>

## Overview

A clip-level deepfake detector with spatial, FFT, audio and across-frame representations. The thesis evaluates saved checkpoints on a 16-clip test split from a 110-clip WildDeepfake-derived collection.

**Author:** MD RAKIBUL ISLAM RAIHAN — M.Eng. Software Engineering, Northwestern Polytechnical University (985/211), GPA 88/100 (Top 1%)
- Email: raihanrakib.143@gmail.com
- GitHub: [theraihanrakibb](https://github.com/theraihanrakibb)
- Portfolio: [theraihanrakibb.github.io](https://theraihanrakibb.github.io/)
**Supervisor:** ZHANG PENG

**University:** Northwestern Polytechnical University, School of Software Engineering

---

[Download the thesis PDF](docs/thesis.pdf)

## WildDeepfake held-out results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multimodal detector | 50.00% | 50.00% | 100.00% | 66.67% | 0.4844 |
| MesoNet | 56.25% | 55.56% | 62.50% | 58.82% | 0.6250 |
| XceptionNet | 56.25% | 54.55% | 75.00% | 63.16% | 0.7969 |
| LipSyncDetector | 43.75% | 0.00% | 0.00% | 0.00% | 0.7969 |
| Equal-weight ensemble | 75.00% | 70.00% | 87.50% | 77.78% | 0.8281 |

The split uses 77 training clips, 17 validation clips and 16 test clips. The training clips have no audio.

## Reproduce the thesis results

From the repository root, audit your saved score files without a GPU:

```bash
python scripts/audit_saved_evaluation.py \
  path/to/held_out_test.json \
  path/to/digifakeav_official_examples.json
```

The ensemble averages the four clip scores. A mean above 0.5 is classified as fake. The proposed detector combines spatial and frequency features, order-independent frame context and an audio input branch. The DigiFakeAV run uses ten publisher examples to check audio-bearing inputs with the same saved checkpoints. Its results apply to those examples.

## Run another experiment

The Linux runners expect a Python virtual environment at `.venv`. The reported training run used Python 3.11 and PyTorch 2.7.1 with its CUDA 12.6 build on an RTX 6000 Ada. Install the packages from `requirements.txt` into the environment.

```bash
bash scripts/run_wilddeepfake_held_out.sh
bash scripts/run_digifakeav_examples.sh results/YYYY-MM-DD/wilddeepfake_held_out/run_HH-MM-SS
bash scripts/run_unit_tests.sh
```

Use the fresh WildDeepfake run path from the first command for the second command. The run directories hold the split manifest, loss histories and scores. The matching logs are under `logs/`.

## Project files

- `src/` contains the models, training code, media processing and evaluation tools.
- `scripts/` contains the recorded experiment entry points and the metric audit.
- `data/wilddeepfake/` is the local 110-clip collection used for the thesis. `data/digifakeav_official_examples/` contains the separate audio-bearing sample.
- `results/` contains saved experiment records and outputs.
- `tests/` contains unit tests and small media fixtures.


## Citation and license

See [CITATION.cff](CITATION.cff) for attribution and [LICENSE](LICENSE) for the software license.

## Related Research

- **B.Eng. Thesis:** [Design and Implementation of a Distributed Confidential Query Protocol for Spark](https://github.com/theraihanrakibb/spark-ckks-query) — Apache Spark + CKKS homomorphic encryption.
- **NVFP4-DiT:** [4-bit Audio-Guided Video Diffusion Transformers](https://github.com/theraihanrakibb/nvfp4-dit) (IEEE TNNLS, under review).
- **Profile & Portfolio:** [@theraihanrakibb](https://github.com/theraihanrakibb) · [Online Portfolio](https://theraihanrakibb.github.io/)

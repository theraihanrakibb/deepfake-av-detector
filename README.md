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

**Author:** [MD Rakibul Islam Raihan](https://github.com/theraihanrakibb)<br>
**Supervisor:** ZHANG PENG<br>
**University:** Northwestern Polytechnical University, School of Software Engineering

The detector combines RGB and log-FFT features, audio features, and a position-aware Transformer. The thesis compares an initial and an improved protocol on the same held-out clips within each dataset. The 110-clip WildDeepfake collection uses a silence fallback. The 300-clip DigiFakeAV subset contains audio.

## Verified results

Each reported full experiment uses four seeds and a 20-epoch budget. Experiment 1 selects checkpoints by validation loss and uses equal ensemble weights. Experiment 2 uses pretrained initialisation, validation-AUC checkpoint selection, validation-derived cutoffs, and validation-weighted ensembling.

- **WildDeepfake:** Multimodal mean ROC AUC rose from 0.4844 to 0.8242. Ensemble mean AUC rose from 0.6641 to 0.8398.
- **DigiFakeAV:** Multimodal mean ROC AUC rose from 0.5300 to 0.7239. Ensemble mean AUC rose from 0.6894 to 0.7189.

The results describe the evaluated subsets. Frame shuffling preserved AUC in every improved seed on both datasets, and zeroing the DigiFakeAV audio features changed the mean AUC from 0.7239 to 0.7244. The ablation series uses separately fitted configurations.

- [Named thesis PDF](results/ddv_avti.pdf)

## Run the code

The Linux runners expect a Python environment at `.venv`, four NVIDIA GPUs, and local dataset files, split manifests, and cached pretrained weights. The full runs used PyTorch 2.7.1 with CUDA 12.6 on RTX 6000 Ada GPUs. Install the matching PyTorch wheels and the packages in [requirements.txt](requirements.txt).

The runners expect WildDeepfake data under `data/wilddeepfake/` and DigiFakeAV data under `data/digifakeav_subset/`. The data, private run logs, and model weights are kept separately. Run the two datasets in separate GPU allocations or one after the other.

```bash
ROOT=results/runs/wilddeepfake bash scripts/run_wilddeepfake.sh
ROOT=results/runs/digifakeav bash scripts/run_digifakeav.sh
bash scripts/run_unit_tests.sh
```

The current runners implement the improved setup. The thesis describes the initial protocol separately. See `src/` for model and evaluation code, `scripts/` for run entry points, and `tests/` for checks.

See [CITATION.cff](CITATION.cff) and [LICENSE](LICENSE).

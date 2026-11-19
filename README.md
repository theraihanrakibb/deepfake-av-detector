# Detecting Deepfake Videos by a Multimodal Audio-Visual Framework with Temporal Inconsistencies

**M.Eng. thesis in Software Engineering, Northwestern Polytechnical University (2026)**

[Read the thesis PDF](results/ddv_avti.pdf)

**Author** [MD Rakibul Islam Raihan](https://github.com/theraihanrakibb)

**Supervisor** Professor Zhang Peng

## Research

The proposed detector assigns a fake score to each video clip. It combines frame appearance, spatial frequency features, audio descriptors, and order-independent frame context. The separate equal-weight ensemble averages the scores of the proposed detector, MesoNet, XceptionNet, and LipSyncDetector.

## Held-out results

The WildDeepfake-derived collection contained 110 clips, with 77 for training, 17 for validation, and 16 for testing. The test split included eight authentic and eight manipulated clips.

| Detector | Correct decisions | Accuracy | ROC AUC |
| --- | ---: | ---: | ---: |
| Proposed multimodal detector | 8/16 | 50.00% | 0.4844 |
| MesoNet | 9/16 | 56.25% | 0.6250 |
| XceptionNet | 9/16 | 56.25% | 0.7969 |
| LipSyncDetector | 7/16 | 43.75% | 0.7969 |
| Equal-weight ensemble | 12/16 | 75.00% | 0.8281 |

The proposed detector flagged every held-out clip as fake at the 0.5 cutoff. The training clips had no audio tracks. Ten publisher-provided DigiFakeAV examples were used separately to check processing of audio-bearing clips. These measurements describe the recorded split and diagnostic sample.

## Repository

- [Thesis PDF](results/ddv_avti.pdf) contains the model specification, dataset split, evaluation and reproduction records.
- [Source code](src/) contains model implementations and experiment utilities.
- [Tests](tests/) contains unit tests and small input fixtures.

The research code is released under the [MIT license](LICENSE).

# Ecuadorian Sign Language — reproducible fruit-sign recognition

[![Tests](https://github.com/andresrubiop/ecsl-sign-recognition/actions/workflows/tests.yml/badge.svg)](https://github.com/andresrubiop/ecsl-sign-recognition/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-e95420.svg)](LICENSE)

A research pipeline for **16 Ecuadorian Sign Language fruit signs**, using MediaPipe landmarks, temporal neural models, CUDA training and **signer-disjoint evaluation**. The package is named `pfrutas`.

This public edition releases **source code, tests and aggregate evaluation**. It does not distribute participant videos, processed landmark sequences, model checkpoints or ASL Citizen research materials.

## Engineering focus

- Presence masks, bounded interpolation and geometric deltas avoid treating missing hands as observed features.
- A compact BiLSTM with attention models local gesture dynamics.
- An optional ST-GCN branch transfers motion representations from ASL Citizen without assuming lexical equivalence between sign languages.
- Nested leave-one-signer-out validation selects epochs on inner folds and evaluates once on the held-out signer.
- PyTorch training/inference uses CUDA in the recommended pipeline; MediaPipe extraction runs on CPU.
- Jetson Orin NX bundle-building and preflight tools are included. Hardware-specific deployment still needs validation on the target device.

## Reported evaluation

| Measure | Historical nested-LOSO result |
| --- | ---: |
| Eligible videos | 72 |
| Signers | 5 |
| Classes | 16 |
| Accuracy | 81.94% |
| Macro F1 | 81.15% |
| Top-3 accuracy | 90.28% |

These values come from a recorded V100 training run, not a new experiment performed for this release. See [aggregate results and provenance](results/nested-loso-summary.json). With five signers and a restricted vocabulary, this is not evidence of general sign-language translation or performance on a new population. Final-refit training accuracy must not be reported as held-out performance.

## Install and run

Use Python **3.10–3.12**. For CPU unit tests, install a CPU build of PyTorch first; for training, choose a compatible CUDA build using the official PyTorch instructions.

```bash
git clone https://github.com/andresrubiop/ecsl-sign-recognition.git
cd ecsl-sign-recognition
python3.12 -m venv .venv
. .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]'
pytest -q
```

With an authorized dataset in `raw/` and a CUDA environment:

```bash
python scripts/00_check_cuda.py
python scripts/01_prepare_dataset.py
python scripts/02_train_nested_cv_gpu.py
python scripts/03_evaluate_oof.py
```

The transfer experiment additionally needs a separately obtained pretrained checkpoint. `scripts/06_download_asl_citizen_weights.py` requires explicit acceptance of the upstream research terms. Nothing is downloaded by installation or unit testing. See the [Spanish pipeline guide](docs/pipeline.es.md), [model card](MODEL_CARD.md) and [distribution policy](MODEL_RELEASE.md).

## Layout

```text
src/pfrutas/   data, features, models, training, live inference and Jetson tools
scripts/       numbered preparation, training, evaluation and deployment entry points
tests/         synthetic fixtures and unit tests
results/       aggregate numerical evaluation, without individual predictions
deploy/        target-device documentation
```

The MIT license covers the released source code; retained Microsoft code notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). It does not grant rights to datasets, pretrained models or derived checkpoints. [Provenance](PROVENANCE.md) identifies the source version and exclusions.

By [Andrés Rubio Proaño](https://www.andresrubiop.com). Research software: not a proficiency assessment, clinical tool or general-purpose translator.

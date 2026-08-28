# NetEvolve

## Project Overview

**NetEvolve** is a research project for **adaptive open-set network traffic classification**.
The long-term goal is a classifier that doesn't just sort traffic into a fixed set of known
categories, but can recognize when it's seeing something it has never been trained on,
group that novel traffic into candidate new categories, and incorporate verified new
categories back into itself over time — without retraining from scratch or forgetting what
it already knew.

The full future vision combines four capabilities:

```
Known Classification  +  Unknown Detection  +  Novel Class Discovery  +  Incremental Learning
```

**This repository, in its current state, implements only the foundation for that vision:**

- Data preparation and validation
- A closed-set baseline classifier (Random Forest)
- A neural network classifier that also learns a reusable embedding space
- Evaluation and comparison of the two

Open-set detection, clustering, human-in-the-loop labeling, and incremental learning are
**not implemented yet** — see [Future Work](#future-work).

---

## Project Objectives

1. Build a reliable known-class network traffic classifier.
2. Learn meaningful deep representations (embeddings) of network traffic.
3. Create a reproducible, leakage-free preprocessing pipeline.
4. Prepare an embedding architecture that later phases can reuse for unknown-traffic
   discovery, without having to retrain the base model.

---

## Dataset

This project uses **UNSW-NB15**, a labeled network traffic dataset containing both normal
traffic and multiple categories of attack traffic, with flow-level features (connection
info, packet/byte counts, timing, and behavioral statistics).

- The multi-class target column is `attack_cat` — this is what the models in this phase
  are trained to predict.
- The binary column `label` (normal vs. attack) exists in the dataset but is **not** used
  as the training target in this phase.
- This phase trains only on a configured subset of categories ("known classes"). The
  remaining categories are deliberately withheld to later evaluate open-set / novel-class
  behavior — they are **not seen during training in this phase.**

Configured for this phase (`config/classes.yaml`):

| Known classes (used for training) | Future unknown classes (withheld) |
|---|---|
| Normal | Analysis |
| DoS | Backdoors |
| Exploits | Reconnaissance |
| Fuzzers | Shellcode |
| Generic | Worms |

> Exact category spelling will be validated against the real dataset values during
> preprocessing (Phase 3) — `label_processor.py` will flag any configured class name that
> matches zero rows.

---

## Architecture

```
UNSW-NB15 Dataset
        |
Data Validation
        |
Data Cleaning
        |
Class Configuration (known vs. future-unknown)
        |
Feature Processing (ColumnTransformer: StandardScaler + OneHotEncoder)
        |
Train / Validation / Test Split (stratified, fit only on train)
        |
        +----------------------+
        |                      |
Random Forest           Neural Network
Baseline                Feature Extractor
        |                      |
        +----------+-----------+
                   |
              Evaluation
                   |
             Saved Models
```

---

## Folder Structure

```
NetEvolve/
├── README.md              This file
├── requirements.txt        Pinned Python dependencies
├── .gitignore
├── pyproject.toml          black/flake8/pytest configuration
├── config/
│   ├── config.yaml          Paths, hyperparameters, training settings
│   └── classes.yaml         Known vs. future-unknown class lists (single source of truth)
├── data/
│   ├── raw/                 Original CSVs — never modified (git-ignored)
│   ├── interim/              Intermediate cleaned data (git-ignored)
│   └── processed/            Final ML-ready arrays + metadata (git-ignored)
├── notebooks/                Exploration only — calls into src/, no pipeline logic
├── src/                      Reusable, tested production code
│   ├── data/                 loading, validation, PyTorch Dataset
│   ├── preprocessing/        cleaning, class filtering, feature processing
│   ├── models/               baseline, neural network, trainer, factory
│   ├── evaluation/           metrics, evaluator, plotting
│   └── utils/                logging, seeding, path/config resolution
├── scripts/                  One CLI entry point per pipeline stage
├── models/                   Saved trained models (git-ignored)
├── results/
│   ├── figures/               Plots
│   ├── metrics/                JSON metric dumps
│   ├── reports/                 Validation / class-configuration reports
│   └── logs/                    Runtime logs (git-ignored)
└── tests/                    pytest unit tests
```

---

## Installation

All commands below are for **Windows (PowerShell or cmd)**.

```powershell
git clone <your-repo-url> NetEvolve
cd NetEvolve

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Setup

Download the UNSW-NB15 CSV files and place them here manually:

```
data/raw/UNSW_NB15_training-set.csv
data/raw/UNSW_NB15_testing-set.csv
```

`src/data/loader.py` (Phase 2) will check for these files explicitly and raise a clear
error — not a silent failure — if either is missing.

---

## Configuration

- **`config/config.yaml`** — dataset paths, random seed, preprocessing settings, baseline
  and neural network hyperparameters, and training settings (batch size, epochs, learning
  rate, early stopping patience, etc.).
- **`config/classes.yaml`** — the known-class and future-unknown-class lists. This is the
  only place class names should be edited; no file under `src/` should hardcode a class
  list.

To change which categories are treated as "known" for a given experiment, edit
`config/classes.yaml` only.

---

## Running the Pipeline

The following pipeline stages are fully implemented and can be run sequentially:

```powershell
python scripts/run_data_validation.py
python scripts/run_preprocessing.py
python scripts/train_baseline.py
python scripts/train_neural_model.py
python scripts/evaluate_model.py
```

---

## Models

- **Random Forest baseline** (`src/models/baseline.py`) — establishes a reference score
  before investing in the neural network.
- **Neural network** (`src/models/neural_network.py`) — a feed-forward classifier over an
  explicit embedding layer (`model.extract_embeddings(x)`), sized 64-dim by default. The
  embeddings themselves aren't used for anything in this phase, but the interface exists
  now so later phases (clustering, novel-class discovery) don't require retraining.

---

## Evaluation Metrics

Accuracy alone is not reported as sufficient given expected class imbalance. Every
evaluation reports:

- Accuracy
- Precision / Recall (macro and weighted)
- Macro F1 *(primary model-selection metric)*
- Weighted F1
- Per-class Precision / Recall / F1
- Confusion matrix

---

## Results

Baseline and neural network performance metrics are generated at runtime and saved to `results/metrics/` and `results/figures/`.
*(Full cross-validation and hyperparameter tuning results will be documented here once finalizing the closed-set benchmarks).*

---

## Current Scope

**Completed up to Phase 5 (Closed-Set Foundation)**: 
- Project structure, configuration, and comprehensive unit tests.
- Leak-free data validation and preprocessing pipeline (StandardScaler + OneHotEncoder).
- Random Forest Baseline and PyTorch Neural Network (with embedding extraction).
- Complete training, early stopping, and evaluation loops.
- Core pipeline scripts fully integrated and executable.

The repository is now ready to begin work on Phase 6 & 7 (Open-Set Recognition, Novel Class Discovery, and Incremental Learning).

---

## Future Work

- Open-set recognition
- Confidence-based uncertainty estimation
- Unknown traffic detection
- Embedding-space analysis
- HDBSCAN clustering for novel class discovery
- UMAP visualization
- Human/admin verification workflow
- Incremental / continual learning with replay buffers
- Streamlit monitoring dashboard
- Multi-dataset evaluation

---

## Reproducibility

- A single random seed (`config/config.yaml -> project.random_seed`) is applied to
  Python's `random`, NumPy, and PyTorch.
- All preprocessing objects (scaler/encoder, label encoder) are fit **only** on the
  training split and saved with `joblib` so the exact same transformation can be re-applied.
- Trained models, their configs, and evaluation metrics are all saved together with a
  metadata file describing how they were produced.

---

## Limitations

- This version uses engineered, flow-level features from UNSW-NB15 — it does not
  reproduce a packet-level or multi-view architecture.
- No open-set, clustering, or incremental-learning capability exists yet; this repository
  is a foundation for those future phases, not a complete adaptive system.

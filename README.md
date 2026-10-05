# GraPEX: GRAdual Pattern EXplainer for Multivariate Time Series Classification

Official code for the paper **"GraPEX: GRAdual Pattern EXplainer for multivariate time series classification"**, accepted at the *ICDM 2026 Workshop on Trustworthy Machine Learning for Fair, Private, Robust, and Explainable Decision-Making (TML4DM)*.


---

GraPEX is a model-agnostic, post-hoc **counterfactual** explainer for multivariate time series (MTS) classifiers. It uses **seasonal gradual patterns**, which are co-variation rules such as *"when channel A increases, channel B decreases"*, to produce:

1. a **counterfactual** MTS `X'` that the black box assigns to a chosen target class, and
2. a **symbolic explanation**: the target-class gradual patterns (channels, directions, time window) used to build it.

The method has four stages:

| Stage | What it does |
|-------|--------------|
| 1. Pattern extraction | Mines seasonal gradual patterns for each class on the training set (`σ_min = 0.4`, `ℓ_min = 7`). If no pattern is found, the thresholds are relaxed step by step. Patterns are mined once and reused for every test instance. |
| 2. NUN search | Finds the nearest unlike neighbour (closest training instance predicted as the target class), using Euclidean distance (default) or DTW. |
| 3. Pattern selection | Keeps target-class patterns that the query does not satisfy (C1), the NUN does satisfy (C2), and that raise the target-class confidence by more than τ (C3). |
| 4. Counterfactual construction | Blends the query towards the NUN, only on the pattern channels and time windows, increasing the blending factor α until the prediction flips. |

## Installation

Python 3.12 is required. With [uv](https://github.com/astral-sh/uv):

```bash
git clone git@github.com:kndbvortex/GRaPEX.git
cd GRaPEX
uv sync
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

UEA datasets are downloaded automatically by `aeon` on first use.

## Quick start

```python
import numpy as np
from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder
from grapex import GraPEX

X_train, y_train = load_classification("BasicMotions", split="train")   # (N, C, T)
X_test,  y_test  = load_classification("BasicMotions", split="test")
le = LabelEncoder().fit(y_train)
y_train, y_test = le.transform(y_train), le.transform(y_test)

# `model` is any classifier with predict / predict_proba on arrays of shape (N, C, T)
model = ...

explainer = GraPEX()
explainer.extract_class_patterns(X_train, y_train, min_support=0.4, min_length=7)

x = X_test[0]
pred = int(model.predict(x[np.newaxis])[0])
result = explainer.explain(x, X_train, y_train, predicted_class=pred,
                           target_class=(pred + 1) % len(le.classes_),
                           metric="euclidean", model=model)

result["counterfactual"]      # (C, T) counterfactual
result["valid"]               # True if the model predicts the target class
result["target_patterns"]     # gradual patterns of the target class
```

## Reproducing the paper

All experiments are run with `experiments/run_benchmark.py`.

**Main benchmark** (Tables III, IV and VI): ResNet, 8 UEA datasets, all methods:

```bash
uv run python experiments/run_benchmark.py
```

**Model-agnosticism** (Table V): ROCKET classifier on three datasets:

```bash
uv run python experiments/run_benchmark.py --classifier rocket \
    --dataset Epilepsy Libras NATOPS --methods GraPEX
```

Main options:

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset NAME [NAME ...]` | 8 paper datasets | Datasets to run |
| `--methods M [M ...]` | all | Subset of `GraPEX GraPEX_DTW GraPEX_Relevance CoMTE TSEvo SETS CONFETTI LASTS` |
| `--classifier {resnet,rocket}` | `resnet` | Black-box classifier (CONFETTI and LASTS need ResNet) |
| `--max-instances N` | 5 | Instances per category (correctly classified / misclassified) |
| `--min-support F` | 0.4 | Initial minimum pattern support `σ_min` |
| `--min-length N` | 7 | Initial minimum pattern length `ℓ_min` |
| `--timeout S` | 7200 | Per-method timeout (seconds) |
| `--no-tshap` / `--no-lasts` | off | Skip the TSHAP50 metric / the LASTS baseline |
| `--jobs N` | 1 | Datasets processed in parallel |
| `--cpu` | auto | Force CPU |

Results are written to `results/<classifier>/<dataset>/metrics.csv`, with aggregated CSVs and LaTeX tables in `results/<classifier>/`.

The trained ResNet weights for six of the eight datasets are provided in `models/`. The remaining ones (FingerMovements, HandMovementDirection) and ROCKET models are trained on first run and cached there.

### Datasets

| Domain | Dataset | Length | Dims | Classes | Train |
|--------|---------|-------:|-----:|--------:|------:|
| Medical | AtrialFibrillation | 640 | 2 | 3 | 15 |
| Medical | FingerMovements | 50 | 28 | 2 | 316 |
| Medical | HandMovementDirection | 400 | 10 | 4 | 160 |
| Motion | BasicMotions | 100 | 6 | 4 | 40 |
| Motion | Epilepsy | 207 | 3 | 4 | 137 |
| Motion | Libras | 45 | 2 | 15 | 180 |
| Motion | NATOPS | 51 | 24 | 6 | 180 |
| Motion | RacketSports | 30 | 6 | 4 | 151 |

### Metrics

- **Validity**: the counterfactual is predicted as the target class.
- **L2 proximity**: Euclidean distance between `X` and `X'` (lower is better).
- **Sparsity**: percentage of positions left unmodified, `|X − X'| ≤ 0.01` (higher is better).
- **Plausibility**: mean L2 distance from `X'` to the target-class training instances (lower is better).
- **TSHAP50**: percentage of modified positions that fall in the top 50% of T-SHAP attributions (higher is better).

## Repository structure

```
grapex/                  GraPEX implementation
  msgp.py                  seasonal gradual pattern mining
  explainer.py             GraPEX explainer (NUN search + pattern-guided interpolation)
  relevance.py             pattern relevance (C1–C3, Δf, score) and relevance-based CF
  nun.py                   nearest unlike neighbour search (Euclidean / DTW)
experiments/
  run_benchmark.py         benchmark reproducing the paper's results
  model_compat.py          loader for Keras 2 / Keras 3 saved models
models/                  trained ResNet weights (one folder per dataset)
third_party/             baselines and evaluation tools (see third_party/README.md)
```

## Citation

If you use this code, please cite:

```bibtex
comming soon
```

## Acknowledgements

We thank IMT Nord Europe and Clermont Auvergne University for their financial support. The baselines in `third_party/` come from their original authors; see `third_party/README.md` for sources and licences and datasets author.

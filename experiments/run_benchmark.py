"""
run_benchmark.py — Counterfactual benchmark of the GraPEX paper (ICDMW 2026).

Methods : GraPEX (Euclidean NUN) · GraPEX_DTW · GraPEX_Relevance
          CoMTE · TSEvo · SETS · CONFETTI · LASTS
Metrics : validity · L2 proximity · sparsity · plausibility · TSHAP50 alignment

Instance selection per dataset:
  • up to --max-instances correctly classified test instances (target = pred + 1)
  • up to --max-instances misclassified test instances      (target = true class)

Output  : results/{CLASSIFIER}/{DATASET}/metrics.csv
          results/{CLASSIFIER}/aggregate_*.csv  +  paper_table_*.tex

Usage:
    python experiments/run_benchmark.py [--dataset NAME ...] [--methods M ...]
                                        [--classifier resnet|rocket] [--cpu]
"""

import argparse
import csv
import os
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import numpy as np

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",      default=None, nargs="+",
                    help="One or more dataset names (default: the 8 paper datasets). "
                         "Example: --dataset BasicMotions Epilepsy")
parser.add_argument("--timeout",      type=int, default=7200,
                    help="Safety timeout per method in seconds (default: 7200). "
                         "Note: Python threads cannot be killed mid-execution; "
                         "this only prevents the main thread from waiting forever "
                         "in case of deadlock or infinite loop.")
parser.add_argument("--cpu",          action="store_true",
                    help="Force CPU (auto-detected if no CUDA GPU found)")
parser.add_argument("--no-tshap",     action="store_true",
                    help="Skip T-SHAP (no tshap_overlap_top50 column)")
parser.add_argument("--classifier",   choices=["resnet", "rocket"], default="resnet",
                    help="Black-box classifier (default: resnet). CONFETTI and LASTS "
                         "require the Keras ResNet and are skipped with rocket.")
parser.add_argument("--methods",      default=None, nargs="+",
                    help="Subset of methods to run (default: all). "
                         "Example: --methods GraPEX GraPEX_DTW")
parser.add_argument("--min-support",  type=float, default=0.4)
parser.add_argument("--min-length",   type=int,   default=7)
parser.add_argument("--epochs",       type=int,   default=500)
parser.add_argument("--lasts-epochs", type=int,   default=200,
                    help="VAE training epochs for LASTS (default 200; reduce on CPU)")
parser.add_argument("--no-lasts",     action="store_true",
                    help="Skip LASTS (VAE training is slow on CPU-only servers)")
parser.add_argument("--jobs",         type=int,   default=1,
                    help="Number of datasets to run in parallel OS processes "
                         "(default 1). Use e.g. --jobs 6 on a 48-thread server.")
parser.add_argument("--max-instances", type=int,  default=5,
                    help="Max instances to evaluate per category (correct / misclassified) "
                         "per dataset (default 5 → up to 10 instances per dataset).")
args = parser.parse_args()

# Auto-detect: no CUDA GPU → force CPU to avoid TF startup warnings
if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if out.returncode != 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    except Exception:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from aeon.datasets import load_classification

project_root = Path(__file__).resolve().parent.parent
for p in [project_root,
          project_root / "third_party",
          Path(__file__).resolve().parent]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# ── Datasets ─────────────────────────────────────────────────────────────────

PAPER_DATASETS = [
    # Medical
    "AtrialFibrillation", "FingerMovements", "HandMovementDirection",
    # Motion
    "BasicMotions", "Epilepsy", "Libras", "NATOPS", "RacketSports",
]

DATASETS       = args.dataset if args.dataset else PAPER_DATASETS
METHOD_TIMEOUT = args.timeout

FIELDNAMES = [
    "dataset", "method", "instance_idx",
    "true_class", "pred_class", "target_class",
    "correctly_classified",        # True = correct, False = misclassified
    "execution_time",
    "valid", "l2_distance", "sparsity_pct", "plausibility",
    "tshap_overlap_top50", "cf_pred_class",
]

# ── Output dirs ───────────────────────────────────────────────────────────────

OUT_ROOT   = project_root / "results" / args.classifier
OUT_ROOT.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_ROOT / "summary.csv"

_csv_lock = threading.Lock()


def _append_row(csv_path: Path, row: dict) -> None:
    with _csv_lock:
        exists = csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})


# ── Metrics ───────────────────────────────────────────────────────────────────

def _plausibility(cf: np.ndarray, X_train: np.ndarray,
                  y_train: np.ndarray, target_class: int) -> float:
    X_tgt = X_train[y_train == target_class]
    if len(X_tgt) == 0:
        return float("nan")
    return float(np.mean([np.linalg.norm(cf - x) for x in X_tgt]))


def _sparsity_pct(original: np.ndarray, cf: np.ndarray,
                  threshold: float = 0.01) -> float:
    """Eq. (6): percentage of (channel × timestep) positions left unmodified.
    Higher = sparser (fewer positions changed)."""
    unchanged = np.sum(np.abs(original - cf) <= threshold)
    return float(unchanged / original.size * 100)


def _tshap_overlap_top50(original: np.ndarray, cf: np.ndarray,
                         tshap_attr: np.ndarray) -> float:
    """
    % of modified positions (|orig−cf| > 0.01) that fall in the top-50%
    most important positions according to |TSHAP|.  Higher = better alignment.
    """
    change_mask = np.abs(original - cf).flatten() > 0.01
    if change_mask.sum() == 0:
        return 0.0
    abs_attr     = np.abs(tshap_attr).flatten()
    threshold    = np.percentile(abs_attr, 50)
    important    = abs_attr >= threshold
    return float(np.sum(change_mask & important) / change_mask.sum() * 100)


def _compute_metrics(original: np.ndarray, cf: np.ndarray, model,
                     true_class: int, target_class: int,
                     X_train: np.ndarray, y_train: np.ndarray,
                     tshap_attr=None) -> dict:
    orig_pred = int(model.predict(original[np.newaxis])[0])
    cf_pred   = int(model.predict(cf[np.newaxis])[0])
    return {
        "valid":              cf_pred == target_class,
        "l2_distance":        float(np.linalg.norm(original - cf)),
        "sparsity_pct":       _sparsity_pct(original, cf),
        "plausibility":       _plausibility(cf, X_train, y_train, target_class),
        "cf_pred_class":      cf_pred,
        "pred_class":         orig_pred,
        "tshap_overlap_top50": (
            _tshap_overlap_top50(original, cf, tshap_attr)
            if tshap_attr is not None else None
        ),
    }


# ── ResNet wrapper ────────────────────────────────────────────────────────────

class ResNetWrapper:
    """Keras ResNet — sklearn interface, aeon format (N, C, T)."""

    def __init__(self, model_dir: Path, nb_classes: int,
                 epochs: int = 500, augment: bool = True):
        self.model_dir  = model_dir
        self.nb_classes = nb_classes
        self.epochs     = epochs
        self.augment    = augment
        self.model      = None
        self.classes_   = np.arange(nb_classes)

    @staticmethod
    def _to_keras(X):               # (N,C,T) → (N,T,C)
        return np.transpose(X, (0, 2, 1))

    def _augment(self, X, y):
        rng = np.random.default_rng(42)
        bX, by = [X], [y]
        for _ in range(5):
            bX.append(X + rng.normal(0, 0.03, X.shape)); by.append(y)
            bX.append(X * rng.uniform(0.9, 1.1, (len(X), 1, 1))); by.append(y)
        Xa = np.concatenate(bX); ya = np.concatenate(by)
        idx = rng.permutation(len(Xa))
        return Xa[idx], ya[idx]

    def _build(self, input_shape):
        import keras
        F   = 64
        inp = keras.layers.Input(input_shape)

        def _res(x_in, filters):
            x  = keras.layers.Conv1D(filters, 8, padding="same")(x_in)
            x  = keras.layers.BatchNormalization()(x)
            x  = keras.layers.Activation("relu")(x)
            x  = keras.layers.Conv1D(filters, 5, padding="same")(x)
            x  = keras.layers.BatchNormalization()(x)
            x  = keras.layers.Activation("relu")(x)
            x  = keras.layers.Conv1D(filters, 3, padding="same")(x)
            x  = keras.layers.BatchNormalization()(x)
            sc = keras.layers.Conv1D(filters, 1, padding="same")(x_in)
            sc = keras.layers.BatchNormalization()(sc)
            return keras.layers.Activation("relu")(keras.layers.add([sc, x]))

        out = _res(inp, F)
        out = _res(out, F * 2)
        out = _res(out, F * 2)
        out = keras.layers.GlobalAveragePooling1D()(out)
        out = keras.layers.Dense(self.nb_classes, activation="softmax")(out)
        m   = keras.models.Model(inp, out)
        m.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
        return m

    def fit(self, X, y):
        import keras
        Xr, yr = (self._augment(X, y)
                  if self.augment and len(X) / self.nb_classes < 30 else (X, y))
        Xk = self._to_keras(Xr)
        self.model = self._build((Xk.shape[1], Xk.shape[2]))
        yoh  = keras.utils.to_categorical(yr, self.nb_classes)
        ckpt = self.model_dir / "best_model.keras"
        bs   = min(16, max(1, len(X) // 10))

        class _Stop(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (logs or {}).get("accuracy", 0) >= 1.0:
                    self.model.stop_training = True

        self.model.fit(
            Xk, yoh, batch_size=bs, epochs=self.epochs, verbose=0,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=50, min_lr=1e-4),
                keras.callbacks.ModelCheckpoint(
                    str(ckpt), monitor="loss", save_best_only=True),
                keras.callbacks.EarlyStopping(
                    monitor="loss", patience=10, restore_best_weights=True),
                _Stop(),
            ])
        self.model = keras.models.load_model(str(ckpt))
        return self

    def load(self):
        try:
            from model_compat import load_model_raw
            self.model = load_model_raw(self.model_dir / "best_model.keras")
        except ImportError:
            import keras
            self.model = keras.models.load_model(
                str(self.model_dir / "best_model.keras"))
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(self._to_keras(X), verbose=0), axis=1)

    def predict_proba(self, X):
        return self.model.predict(self._to_keras(X), verbose=0)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


# ── ROCKET wrapper ────────────────────────────────────────────────────────────

class RocketWrapper:
    """aeon RocketClassifier — sklearn interface, aeon format (N, C, T)."""

    def __init__(self, model_path: Path, nb_classes: int):
        self.model_path = model_path
        self.nb_classes = nb_classes
        self.model      = None
        self.classes_   = np.arange(nb_classes)

    def fit(self, X, y):
        import joblib
        from aeon.classification.convolution_based import RocketClassifier
        self.model = RocketClassifier(random_state=42)
        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)
        return self

    def load(self):
        import joblib
        self.model = joblib.load(self.model_path)
        return self

    def predict(self, X):
        return self.model.predict(X).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


# ── T-SHAP ────────────────────────────────────────────────────────────────────

def _compute_tshap(instance, model, X_train, pred_class):
    """Return (C, T) T-SHAP attributions, or None on failure."""
    try:
        from tshap import TSHAPExplainer
        tshap = TSHAPExplainer(
            window_length=min(20, X_train.shape[2] // 4),
            stride=5, interpolation=True, roi=True,
        )
        baselines = X_train[
            np.random.choice(len(X_train), min(10, len(X_train)), replace=False)]
        window_exp, _ = tshap.explain(
            instance.reshape(1, instance.shape[0], instance.shape[1]),
            baselines, model, clf_targets=np.array([pred_class]),
        )
        return window_exp[0]
    except Exception as exc:
        print(f"      [T-SHAP] failed: {exc}")
        return None


# ── GraPEX helpers ────────────────────────────────────────────────────────────

def _build_grapex(X_train, y_train, min_support, min_length):
    """Mine class patterns once per dataset, with progressive parameter relaxation."""
    from grapex import GraPEX

    def _has_patterns(gp):
        return any(len(p) > 0 for p in gp.class_patterns.values())

    # Progressive fallback: relax constraints until patterns are found
    candidates = [
        (min_support,  min_length),   # user params (default 0.4 / 7)
        (0.3,          5),
        (0.2,          3),
        (0.1,          3),
    ]
    # Deduplicate while preserving order
    search = list(dict.fromkeys(candidates))

    t0 = time.time()
    for sup, length in search:
        gp = GraPEX()
        gp.extract_class_patterns(X_train, y_train,
                                  min_support=sup, min_length=length)
        if _has_patterns(gp):
            elapsed = time.time() - t0
            if (sup, length) != (min_support, min_length):
                print(f"  GraPEX: patterns found with relaxed params "
                      f"(min_support={sup}, min_length={length})")
            return gp, elapsed

    # Last resort: return whatever we got even if empty
    print("  GraPEX: no patterns found even with min_support=0.1 — "
          "GraPEX variants will produce no CF")
    return gp, time.time() - t0


def _grapex_explain(gp, instance, X_train, y_train, pred_class, target_class,
                    metric, model):
    result = gp.explain(instance, X_train, y_train, pred_class, target_class,
                        metric=metric, alpha=0.7, model=model, optimize=True)
    if result.get("no_patterns", False):
        return None
    return result["counterfactual"]


def _run_grapex_relevance(instance, target_class, X_train, y_train_enc, model, gp,
                          k: int = 3, tau: float = 0.01):
    from grapex import relevance_counterfactual
    return relevance_counterfactual(
        instance, target_class, X_train, y_train_enc, model,
        gp.class_patterns.get(target_class, []), k=k, tau=tau)

# ── LASTS dataset-level state ─────────────────────────────────────────────────

class LastsState:
    def __init__(self, encoder, decoder_wrapper, blackbox, class_names):
        self.encoder         = encoder
        self.decoder_wrapper = decoder_wrapper
        self.blackbox        = blackbox
        self.class_names     = class_names


def _build_lasts_state(X_train, model, class_names):
    try:
        from lasts.autoencoders.variational_autoencoder_v2 import build_vae
        from lasts.wrappers import KerasClassifierWrapper, DecoderWrapper

        X_tc        = np.transpose(X_train, (0, 2, 1))   # (N, T, C)
        input_shape = X_tc.shape[1:]
        latent_dim  = min(2, input_shape[1])
        ae_kwargs   = {
            "filters": [4, 4, 4, 4], "kernel_size": [3, 3, 3, 3],
            "padding": ["same"] * 4, "activation": ["relu"] * 4,
            "pooling": [1, 1, 1, 1], "n_layers": 4, "optimizer": "adam",
            "n_layers_residual": None, "batch_normalization": None,
            "kl_weight": 0.1,
        }
        print("  [LASTS] Training VAE...")
        encoder, decoder, autoencoder = build_vae(
            input_shape, latent_dim, ae_kwargs, verbose=False)
        autoencoder.fit(X_tc, X_tc, epochs=args.lasts_epochs, batch_size=16, verbose=0)
        print("  [LASTS] VAE ready.")
        return LastsState(encoder, DecoderWrapper(decoder),
                          KerasClassifierWrapper(model.model), class_names)
    except Exception as exc:
        print(f"  [LASTS] Setup failed: {exc}")
        return None


# ── Per-method runners ────────────────────────────────────────────────────────

def _run_comte(instance, class_label, target_class, X_train, y_train_enc, model):
    from TSInterpret.InterpretabilityModels.counterfactual.COMTECF import COMTECF
    comte = COMTECF(model, data=(X_train, y_train_enc), backend="SK", mode="feat",
                    method="opt", number_distractors=3,
                    max_attempts=1000, max_iter=1000, silent=True)
    cf = comte.explain(instance[np.newaxis], orig_class=class_label, target=target_class)
    if cf is None:
        return None
    if isinstance(cf, tuple):
        cf = cf[0]
    return cf[0] if cf.ndim == 3 else cf


def _run_tsevo(instance, class_label, target_class, X_train, y_train_enc, model):
    from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
    tsevo = TSEvo(model=model, data=(X_train, y_train_enc),
                  backend="SK", mode="feat", epochs=100, verbose=0)
    cf = tsevo.explain(instance[np.newaxis],
                       original_y=class_label, target_y=target_class)
    if cf is None:
        return None
    if isinstance(cf, tuple):
        cf = cf[0]
    return cf[0] if cf.ndim == 3 else cf


def _run_sets(instance, class_label, target_class, X_train, y_train_enc, model):
    from TSInterpret.InterpretabilityModels.counterfactual.SETSCF import SETSCF
    sets = SETSCF(
        model, data=(X_train, y_train_enc), mode="feat", backend="SK",
        min_shapelet_len=3,
        max_shapelet_len=min(20, X_train.shape[2] // 2),
        time_contract_in_mins_per_dim=0.25, silent=True,
    )
    cf = sets.explain(instance, orig_class=class_label, target=target_class)
    if cf is None:
        return None
    if isinstance(cf, tuple):
        cf = cf[0]
    return cf[0] if cf.ndim == 3 else cf


def _run_confetti(instance, class_label, target_class, X_train, model_path):
    from confetti.explainer import CONFETTI
    explainer        = CONFETTI(model_path=str(model_path))
    inst_confetti    = np.transpose(instance, (1, 0))[np.newaxis]   # (1,T,C)
    Xtrain_confetti  = np.transpose(X_train, (0, 2, 1))             # (N,T,C)
    results = explainer.generate_counterfactuals(
        instances_to_explain=inst_confetti,
        reference_data=Xtrain_confetti,
        reference_weights=None,
        alpha=0.5, theta=0.51, n_partitions=3,
        population_size=50, maximum_number_of_generations=50,
        optimize_confidence=True, optimize_sparsity=True,
        optimize_proximity=True, verbose=False,
    )
    if not results or results[0].best is None:
        return None
    cf = results[0].best.counterfactual
    if cf.ndim == 3:
        cf = cf[0]
    return np.transpose(cf, (1, 0))   # (C, T)


def _run_lasts(instance, class_label, target_class,
               lasts_state: LastsState, X_train) -> np.ndarray | None:
    try:
        from lasts.explainers.lasts import Lasts
        from lasts.neighgen.counter_generator import CounterGenerator
        from lasts.utils import choose_z

        instance_tc = np.transpose(instance, (1, 0))[np.newaxis]   # (1,T,C)
        z_fixed = choose_z(
            instance_tc, lasts_state.encoder, lasts_state.decoder_wrapper,
            n=500, x_label=class_label,
            blackbox=lasts_state.blackbox, check_label=True,
        )
        neighgen = CounterGenerator(
            blackbox=lasts_state.blackbox,
            decoder=lasts_state.decoder_wrapper,
            n_search=20000, n_batch=1000,
            lower_threshold=0, upper_threshold=20,
            kind="gaussian_matched", sampling_kind="uniform_sphere",
            verbose=False, n=500, balance=False,
            redo_search=True, cut_radius=True,
        )
        explainer = Lasts(
            blackbox=lasts_state.blackbox,
            encoder=lasts_state.encoder,
            decoder=lasts_state.decoder_wrapper,
            neighgen=neighgen, surrogate=None,
            labels=list(lasts_state.class_names.values()), verbose=False,
        )
        explainer.fit_partial(instance_tc, z_fixed)
        z_cf = neighgen.closest_counterexemplar_
        if z_cf is None:
            return None
        cf_tc = lasts_state.decoder_wrapper.predict(z_cf)  # (1,T,C)
        return np.transpose(cf_tc[0], (1, 0))               # (C,T)
    except Exception as exc:
        print(f"      [LASTS inner] {exc}")
        return None


# ── Dispatcher ────────────────────────────────────────────────────────────────

def _dispatch(name, fn, instance, true_class, target_class,
              X_train, y_train_enc, model,
              dataset_csv, dataset, instance_idx, pred_class,
              correctly_classified, tshap_attr, extra_time: float = 0.0):
    t0 = time.time()
    base = {
        "dataset": dataset, "method": name,
        "instance_idx": instance_idx,
        "true_class": int(true_class), "pred_class": int(pred_class),
        "target_class": int(target_class),
        "correctly_classified": correctly_classified,
    }
    try:
        cf      = fn()
        elapsed = time.time() - t0 + extra_time

        if cf is None:
            row = {**base, "execution_time": f"{elapsed:.2f}", "valid": False}
            _append_row(dataset_csv, row)
            _append_row(SUMMARY_CSV, row)
            print(f"      [{name}] no CF  ({elapsed:.1f}s)")
            return row

        if cf.ndim == 3:
            cf = cf[0]

        m = _compute_metrics(instance, cf, model, true_class, target_class,
                             X_train, y_train_enc, tshap_attr)
        row = {
            **base,
            "execution_time": f"{elapsed:.2f}",
            **{k: (f"{v:.4f}" if isinstance(v, float) else v)
               for k, v in m.items()},
        }
        _append_row(dataset_csv, row)
        _append_row(SUMMARY_CSV, row)
        ts_str = (f"  tshap50={m['tshap_overlap_top50']:.1f}%"
                  if m["tshap_overlap_top50"] is not None else "")
        print(f"      [{name}] valid={m['valid']}  "
              f"L2={m['l2_distance']:.3f}  "
              f"spar={m['sparsity_pct']:.1f}%  "
              f"plaus={m['plausibility']:.3f}{ts_str}  ({elapsed:.1f}s)")
        return row

    except Exception as exc:
        elapsed = time.time() - t0 + extra_time
        row = {**base, "execution_time": f"{elapsed:.2f}",
               "valid": False, "error": str(exc)[:120]}
        _append_row(dataset_csv, row)
        _append_row(SUMMARY_CSV, row)
        print(f"      [{name}] ERROR: {exc}  ({elapsed:.1f}s)")
        return row


# ── Instance driver ───────────────────────────────────────────────────────────

def process_instance(instance_idx, instance, true_class, pred_class,
                     target_class, correctly_classified,
                     X_train, y_train_enc, model, model_path,
                     grapex, grapex_time, lasts_state,
                     dataset_csv, dataset_name,
                     done_pairs: set = None):

    # T-SHAP once per instance
    tshap_attr = None
    if not args.no_tshap:
        print(f"      [T-SHAP] computing...")
        tshap_attr = _compute_tshap(instance, model, X_train, pred_class)

    methods = {
        "CoMTE": lambda: _run_comte(
            instance, pred_class, target_class, X_train, y_train_enc, model),
        "TSEvo": lambda: _run_tsevo(
            instance, pred_class, target_class, X_train, y_train_enc, model),
        "SETS": lambda: _run_sets(
            instance, pred_class, target_class, X_train, y_train_enc, model),
    }

    # CONFETTI and LASTS need the Keras ResNet internals
    if args.classifier == "resnet":
        methods["CONFETTI"] = lambda: _run_confetti(
            instance, pred_class, target_class, X_train, model_path)
    if lasts_state is not None:
        methods["LASTS"] = lambda: _run_lasts(
            instance, pred_class, target_class, lasts_state, X_train)

    # extra_time[method] = overhead to add to execution_time (pattern extraction)
    extra_times: dict = {}

    if grapex is not None:
        for mname, metric in (("GraPEX", "euclidean"), ("GraPEX_DTW", "dtw")):
            methods[mname] = (
                lambda m=metric: _grapex_explain(
                    grapex, instance, X_train, y_train_enc, pred_class,
                    target_class, m, model)
            )
            extra_times[mname] = grapex_time   # amortised extraction overhead

        methods["GraPEX_Relevance"] = lambda: _run_grapex_relevance(
            instance, target_class, X_train, y_train_enc, model, grapex)
        extra_times["GraPEX_Relevance"] = grapex_time

    if args.methods:
        methods = {n: f for n, f in methods.items() if n in args.methods}

    # Resume: skip methods already computed for this instance
    if done_pairs:
        skipped = [n for n in list(methods.keys()) if (instance_idx, n) in done_pairs]
        for n in skipped:
            print(f"      [{n}] already computed — skipping")
            methods.pop(n)

    if not methods:
        print(f"      All methods already done for instance {instance_idx}")
        return

    # Pre-import TSInterpret modules in the main thread to prevent _ModuleLock
    # deadlocks that occur when SETS and TSEvo import from the same package
    # concurrently in worker threads.
    try:
        from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo as _  # noqa
        from TSInterpret.InterpretabilityModels.counterfactual.SETSCF import SETSCF as _  # noqa
        from TSInterpret.InterpretabilityModels.counterfactual.COMTECF import COMTECF as _  # noqa
    except Exception:
        pass

    with ThreadPoolExecutor(max_workers=len(methods)) as pool:
        futures = {
            pool.submit(
                _dispatch,
                name, fn, instance, true_class, target_class,
                X_train, y_train_enc, model,
                dataset_csv, dataset_name, instance_idx, pred_class,
                correctly_classified, tshap_attr,
                extra_times.get(name, 0.0),   # ← extraction overhead
            ): name
            for name, fn in methods.items()
        }
        for future in futures:
            name = futures[future]
            try:
                future.result(timeout=METHOD_TIMEOUT)
            except FuturesTimeout:
                print(f"      [{name}] TIMEOUT after {METHOD_TIMEOUT}s")
                future.cancel()
            except Exception as exc:
                print(f"      [{name}] unhandled: {exc}")


# ── Instance selection ────────────────────────────────────────────────────────

def _select_instances(y_test_enc, y_pred, nb_classes, max_n=5, rng_seed=42):
    """
    Return up to 2 * max_n instances:
      - up to max_n correctly classified  (y_pred == y_true)
      - up to max_n misclassified         (y_pred != y_true)

    Instances are sampled to spread across classes for diversity.
    For each, target_class = (pred_class + 1) % nb_classes.
    """
    rng = np.random.default_rng(rng_seed)
    selected = []

    def _pick_diverse(indices, n):
        """Pick up to n indices, prioritising class diversity."""
        if len(indices) <= n:
            return indices
        classes = y_test_enc[indices]
        chosen = []
        remaining = list(indices)
        # Round-robin across classes
        class_buckets = {}
        for idx in remaining:
            c = int(y_test_enc[idx])
            class_buckets.setdefault(c, []).append(idx)
        for bucket in class_buckets.values():
            rng.shuffle(bucket)
        class_keys = list(class_buckets.keys())
        rng.shuffle(class_keys)
        i = 0
        while len(chosen) < n:
            key = class_keys[i % len(class_keys)]
            if class_buckets[key]:
                chosen.append(class_buckets[key].pop(0))
            i += 1
            if all(len(b) == 0 for b in class_buckets.values()):
                break
        return np.array(chosen[:n])

    # Correct instances
    correct_idx = np.where(y_pred == y_test_enc)[0]
    if len(correct_idx) > 0:
        picked = _pick_diverse(correct_idx, max_n)
        for idx in picked:
            pred_cls   = int(y_pred[idx])
            true_cls   = int(y_test_enc[idx])
            target_cls = (pred_cls + 1) % nb_classes
            selected.append((int(idx), true_cls, pred_cls, target_cls, True))

    # Misclassified instances
    wrong_idx = np.where(y_pred != y_test_enc)[0]
    if len(wrong_idx) > 0:
        picked = _pick_diverse(wrong_idx, max_n)
        for idx in picked:
            pred_cls   = int(y_pred[idx])
            true_cls   = int(y_test_enc[idx])
            target_cls = true_cls
            selected.append((int(idx), true_cls, pred_cls, target_cls, False))

    return selected   # list of (idx, true_cls, pred_cls, target_cls, correctly_classified)


# ── Dataset driver ────────────────────────────────────────────────────────────

def process_dataset(dataset_name: str) -> None:
    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*70}")
    t_ds = time.time()

    try:
        X_train, y_train = load_classification(dataset_name, split="train")
        X_test,  y_test  = load_classification(dataset_name, split="test")
    except Exception as exc:
        print(f"  Cannot load {dataset_name}: {exc}")
        return

    print(f"  Train {X_train.shape}  Test {X_test.shape}")

    le           = LabelEncoder()
    y_train_enc  = le.fit_transform(y_train)
    y_test_enc   = le.transform(y_test)
    nb_classes   = len(le.classes_)
    class_names  = {i: le.inverse_transform([i])[0] for i in range(nb_classes)}

    ds_dir      = OUT_ROOT / dataset_name
    ds_dir.mkdir(exist_ok=True)
    dataset_csv = ds_dir / "metrics.csv"

    # ── Load / train the black-box classifier ───────────────────────────────
    if args.classifier == "rocket":
        model_path = project_root / "models" / f"rocket_{dataset_name.lower()}.joblib"
        wrapper    = RocketWrapper(model_path, nb_classes)
    else:
        model_dir  = project_root / "models" / f"resnet_{dataset_name.lower()}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "best_model.keras"
        wrapper    = ResNetWrapper(model_dir, nb_classes, epochs=args.epochs)
    if model_path.exists():
        print(f"  Loading {args.classifier}...")
        wrapper.load()
    else:
        print(f"  Training {args.classifier}...")
        wrapper.fit(X_train, y_train_enc)
    print(f"  Accuracy: {wrapper.score(X_test, y_test_enc):.4f}")

    # ── GraPEX patterns (mined once, cached for all test instances) ──────────
    print("  Extracting GraPEX patterns...")
    try:
        grapex, grapex_time = _build_grapex(
            X_train, y_train_enc, args.min_support, args.min_length)
        print(f"  GraPEX: {grapex_time:.1f}s")
    except Exception as exc:
        print(f"  GraPEX failed: {exc}")
        grapex, grapex_time = None, 0.0

    # ── LASTS VAE ────────────────────────────────────────────────────────────
    use_lasts = (not args.no_lasts and args.classifier == "resnet"
                 and (not args.methods or "LASTS" in args.methods))
    lasts_state = _build_lasts_state(X_train, wrapper, class_names) if use_lasts else None

    # ── Resume: load already-computed (instance_idx, method) pairs ───────────
    done_pairs: set = set()
    if dataset_csv.exists():
        try:
            import pandas as pd
            df_done = pd.read_csv(dataset_csv)
            df_done["valid_str"] = df_done["valid"].astype(str).str.lower()
            has_result = df_done["valid_str"].isin(["true", "false"])
            for _, row in df_done[has_result].iterrows():
                done_pairs.add((int(row["instance_idx"]), str(row["method"])))
            if done_pairs:
                print(f"  Resume: {len(done_pairs)} (instance, method) pairs already done")
        except Exception as exc:
            print(f"  Resume check failed (will rerun all): {exc}")

    # ── Select instances ──────────────────────────────────────────────────────
    y_pred   = wrapper.predict(X_test)
    selected = _select_instances(y_test_enc, y_pred, nb_classes,
                                 max_n=args.max_instances)
    print(f"  Selected {len(selected)} instance(s) "
          f"(max {args.max_instances} per category):")
    for idx, tc, pc, tgt, corr in selected:
        label = "correct" if corr else "misclassified"
        print(f"    idx={idx}  true={class_names[tc]}  "
              f"pred={class_names[pc]}  target={class_names[tgt]}  [{label}]")

    # ── Run methods per instance ──────────────────────────────────────────────
    for idx, true_cls, pred_cls, target_cls, corr in tqdm(selected,
                                                           desc=f"  {dataset_name}"):
        instance = X_test[idx]
        print(f"\n    Instance {idx} | "
              f"pred={class_names[pred_cls]} → target={class_names[target_cls]}")
        process_instance(
            idx, instance, true_cls, pred_cls, target_cls, corr,
            X_train, y_train_enc, wrapper, model_path,
            grapex, grapex_time, lasts_state,
            dataset_csv, dataset_name,
            done_pairs=done_pairs,
        )

    print(f"\n  {dataset_name} done in {time.time()-t_ds:.1f}s")


# ── Summary & LaTeX ───────────────────────────────────────────────────────────

METHOD_DISPLAY = {
    "GraPEX":                  r"\textsc{GraPEX}",
    "GraPEX_DTW":              r"\textsc{GraPEX-DTW}",
    "GraPEX_Relevance":        r"\textsc{GraPEX-Relevance}",
    "CoMTE":                   r"\textsc{CoMTE}",
    "CONFETTI":                r"\textsc{Confetti}",
    "SETS":                    r"\textsc{Sets}",
    "TSEvo":                   r"\textsc{TSEvo}",
    "LASTS":                   r"\textsc{Lasts}",
}

METHOD_ORDER = [
    "GraPEX", "GraPEX_DTW", "GraPEX_Relevance",
    "CoMTE", "CONFETTI", "SETS", "TSEvo", "LASTS",
]


def compile_summary() -> None:
    import pandas as pd

    dfs = []
    for ds in DATASETS:
        p = OUT_ROOT / ds / "metrics.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["dataset"] = ds
            dfs.append(df)
    if not dfs:
        print("No results to summarise.")
        return

    data = pd.concat(dfs, ignore_index=True)
    for col in ["l2_distance", "sparsity_pct", "plausibility",
                "tshap_overlap_top50", "execution_time"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data["valid"] = data["valid"].map(
        lambda v: True if str(v).lower() in ("true", "1") else False)

    # Quality metrics (L2, sparsity, plausibility, tshap) are only meaningful
    # for valid CFs — mask them out for invalid rows before aggregating.
    valid_mask = data["valid"]
    for col in ["l2_distance", "sparsity_pct", "plausibility", "tshap_overlap_top50"]:
        if col in data.columns:
            data.loc[~valid_mask, col] = float("nan")

    agg = (data.groupby("method")
           .agg(
               validity_pct      =("valid",               lambda s: s.mean() * 100),
               l2_mean           =("l2_distance",         "mean"),
               l2_std            =("l2_distance",         "std"),
               sparsity_mean     =("sparsity_pct",        "mean"),
               sparsity_std      =("sparsity_pct",        "std"),
               plausibility_mean =("plausibility",        "mean"),
               plausibility_std  =("plausibility",        "std"),
               tshap50_mean      =("tshap_overlap_top50", "mean"),
               tshap50_std       =("tshap_overlap_top50", "std"),
               time_mean         =("execution_time",      "mean"),
               n_instances       =("valid",               "count"),
           )
           .reset_index()
           .sort_values("validity_pct", ascending=False))

    agg.to_csv(OUT_ROOT / "aggregate_by_method.csv", index=False)

    per_ds = (data.groupby(["dataset", "method"])
              .agg(
                  validity_pct      =("valid",               lambda s: s.mean() * 100),
                  l2_mean           =("l2_distance",         "mean"),
                  sparsity_mean     =("sparsity_pct",        "mean"),
                  plausibility_mean =("plausibility",        "mean"),
                  tshap50_mean      =("tshap_overlap_top50", "mean"),
                  n                 =("valid",               "count"),
              )
              .reset_index())
    per_ds.to_csv(OUT_ROOT / "aggregate_by_dataset_method.csv", index=False)

    _write_latex_table(agg, per_ds, data["dataset"].nunique())
    print(f"\n{'='*70}")
    print(agg[["method", "validity_pct", "l2_mean",
               "sparsity_mean", "plausibility_mean", "tshap50_mean"]
              ].to_string(index=False))


# ── LaTeX helpers ─────────────────────────────────────────────────────────────

def _fmt(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    return f"{v:.{d}f}"


def _bold_best(vals, higher_is_better):
    finite = [v for v in vals if v is not None and not np.isnan(v)]
    if not finite:
        return [_fmt(v) for v in vals]
    best = max(finite) if higher_is_better else min(finite)
    out = []
    for v in vals:
        s = _fmt(v)
        if v is not None and not np.isnan(v) and abs(v - best) < 1e-9:
            s = r"\textbf{" + s + r"}"
        out.append(s)
    return out


def _write_latex_table(agg, per_ds, n_datasets):
    order   = [m for m in METHOD_ORDER if m in agg["method"].values]
    order  += sorted(m for m in agg["method"].values if m not in order)
    agg_ord = agg.set_index("method").loc[order].reset_index()

    methods = list(agg_ord["method"])
    disp    = [METHOD_DISPLAY.get(m, r"\textsc{" + m + r"}") for m in methods]

    def gcol(col): return list(agg_ord[col]) if col in agg_ord.columns else [np.nan]*len(methods)

    val_f = _bold_best(gcol("validity_pct"),      True)
    l2_f  = _bold_best(gcol("l2_mean"),            False)
    sp_f  = _bold_best(gcol("sparsity_mean"),      True)    # higher = sparser = better
    pl_f  = _bold_best(gcol("plausibility_mean"),  False)
    ts_f  = _bold_best(gcol("tshap50_mean"),       True)
    has_ts = "tshap50_mean" in agg_ord.columns

    # ── Table 1: aggregate ────────────────────────────────────────────────────
    ts_h = r" & TSHAP$_{50}$$\uparrow$" if has_ts else ""
    ts_u = r" & (\%)" if has_ts else ""
    cols = "lrrrr" + ("r" if has_ts else "")
    lines = [
        r"% Auto-generated by run_paper_benchmark.py",
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Counterfactual quality averaged over "
         + str(n_datasets)
         + r" datasets (" + args.classifier + r" classifier). "
         r"$\uparrow$ higher is better; $\downarrow$ lower is better. "
         r"\textbf{Bold} = best per column.}"),
        r"\label{tab:benchmark_aggregate}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{" + cols + r"}",
        r"\toprule",
        (r"Method & Validity$\uparrow$ & L2$\downarrow$ & Sparsity$\uparrow$"
         r" & Plausibility$\downarrow$" + ts_h + r" \\"),
        r"& (\%) & & (\%) & " + ts_u + r" \\",
        r"\midrule",
    ]
    for i in range(len(methods)):
        ts_cell = f" & {ts_f[i]}" if has_ts else ""
        lines.append(
            f"{disp[i]} & {val_f[i]} & {l2_f[i]} & {sp_f[i]} & {pl_f[i]}{ts_cell} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_ROOT / "paper_table_aggregate.tex").write_text("\n".join(lines) + "\n")

    # ── Table 2: per-dataset (validity / sparsity) ────────────────────────────
    methods_pd  = [m for m in order if m in per_ds["method"].values]
    datasets_pd = sorted(per_ds["dataset"].unique())
    disp_pd     = [METHOD_DISPLAY.get(m, m) for m in methods_pd]
    mh = " & ".join(r"\multicolumn{2}{c}{" + d + r"}" for d in disp_pd)
    sh = " & ".join(r"Val & Sp" for _ in methods_pd)
    lines2 = [
        r"% Auto-generated by run_paper_benchmark.py",
        r"\begin{table}[t]", r"\centering",
        r"\caption{Per-dataset results: Validity (\%) / Sparsity (\%).}",
        r"\label{tab:benchmark_per_dataset}", r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l" + "rr" * len(methods_pd) + r"}",
        r"\toprule",
        r"Dataset & " + mh + r" \\",
        r"& " + sh + r" \\", r"\midrule",
    ]
    for ds in datasets_pd:
        cells = []
        for m in methods_pd:
            sub = per_ds[(per_ds["dataset"] == ds) & (per_ds["method"] == m)]
            if sub.empty:
                cells.append("--- & ---")
            else:
                v = sub["validity_pct"].values[0]
                s = sub["sparsity_mean"].values[0]
                cells.append(f"{_fmt(v)} & {_fmt(s)}")
        lines2.append(r"\texttt{" + ds + r"} & " + " & ".join(cells) + r" \\")
    lines2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_ROOT / "paper_table_per_dataset.tex").write_text("\n".join(lines2) + "\n")

    print(f"LaTeX → {OUT_ROOT}/paper_table_aggregate.tex")
    print(f"LaTeX → {OUT_ROOT}/paper_table_per_dataset.tex")


# ── Entry point ───────────────────────────────────────────────────────────────

def _run_dataset_subprocess(ds: str) -> None:
    """Re-invoke this script as a child process for one dataset (true parallelism)."""
    import subprocess
    cmd = [sys.executable, __file__, "--dataset", ds]
    # Forward all relevant flags
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        cmd += ["--cpu"]
    if args.no_tshap:
        cmd += ["--no-tshap"]
    if args.no_lasts:
        cmd += ["--no-lasts"]
    cmd += ["--timeout",       str(args.timeout)]
    cmd += ["--min-support",   str(args.min_support)]
    cmd += ["--min-length",    str(args.min_length)]
    cmd += ["--epochs",        str(args.epochs)]
    cmd += ["--lasts-epochs",  str(args.lasts_epochs)]
    cmd += ["--max-instances", str(args.max_instances)]
    cmd += ["--classifier",    args.classifier]
    if args.methods:
        cmd += ["--methods", *args.methods]
    cmd += ["--jobs", "1"]   # children never spawn sub-children
    print(f"  [launcher] starting subprocess: {ds}")
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        print(f"  [launcher] {ds} exited with code {proc.returncode}")


if __name__ == "__main__":
    print("Datasets :", DATASETS)
    print(f"Classifier: {args.classifier}")
    print(f"Timeout  : {METHOD_TIMEOUT}s / method / instance")
    print(f"T-SHAP   : {'disabled' if args.no_tshap else 'enabled'}")
    print(f"LASTS    : {'disabled' if args.no_lasts else f'enabled ({args.lasts_epochs} VAE epochs)'}")
    print(f"Jobs     : {args.jobs} parallel dataset(s)")
    print("=" * 70)

    if args.jobs > 1:
        # Each dataset runs in its own Python process → no GIL contention
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            list(tqdm(
                pool.map(_run_dataset_subprocess, DATASETS),
                total=len(DATASETS), desc="Datasets", colour="blue",
            ))
    else:
        for ds in tqdm(DATASETS, desc="Datasets", colour="blue"):
            try:
                process_dataset(ds)
            except Exception as exc:
                print(f"\nFatal error on {ds}: {exc}")
                traceback.print_exc()

    print("\nCompiling summary and LaTeX tables...")
    try:
        compile_summary()
    except ImportError:
        print("pandas not available — skipping summary.")
    except Exception as exc:
        print(f"Summary failed: {exc}")
        traceback.print_exc()

    print("\nDone.")

"""Nearest Unlike Neighbour (NUN) search.

Finds the closest training instance of a given target class, using DTW or
Euclidean distance.
"""

import numpy as np
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Pure-NumPy DTW (no numba / tslearn dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _dtw_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    DTW between two 1-D sequences using a vectorised row-by-row DP.
    Returns the normalised cost (divided by n+m).
    Much faster than a pure Python double loop.
    """
    n, m = len(a), len(b)
    # prev row: D[0, 0..m]
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for j in range(1, m + 1):
        prev[j] = np.inf   # D[0, j] = inf (can only start at (0,0))

    # cost matrix column: |a[i-1] - b[j-1]| for all j at once
    for i in range(1, n + 1):
        curr = np.empty(m + 1)
        curr[0] = np.inf
        cost_row = np.abs(a[i - 1] - b)   # shape (m,)
        # curr[j] = cost_row[j-1] + min(prev[j], curr[j-1], prev[j-1])
        # We still need a loop here because curr[j] depends on curr[j-1]
        for j in range(1, m + 1):
            curr[j] = cost_row[j - 1] + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr

    return float(prev[m]) / (n + m)


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Per-channel DTW averaged across channels.

    Parameters
    ----------
    a, b : np.ndarray, shape (C, T)
    """
    n_channels = a.shape[0]
    return float(np.mean([_dtw_1d(a[c], b[c]) for c in range(n_channels)]))


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def find_nun(x: np.ndarray,
             X_train: np.ndarray,
             y_train: np.ndarray,
             model,
             c_target: int,
             metric: str = "dtw",
             min_confidence: float = 0.0) -> Optional[Tuple[np.ndarray, int, float]]:
    """
    Find the Nearest Unlike Neighbor (NUN) for instance x.

    Returns the closest training instance that:
      - belongs to c_target
      - is correctly classified by model (if min_confidence > 0)

    Parameters
    ----------
    x              : np.ndarray, shape (C, T)
    X_train        : np.ndarray, shape (N, C, T)
    y_train        : np.ndarray, shape (N,)
    model          : classifier with predict and (optionally) predict_proba
    c_target       : int — desired target class
    metric         : "dtw" or "euclidean"
    min_confidence : float in [0,1] — minimum predicted probability for c_target
                     to consider an instance as a valid NUN. Set 0 to disable.

    Returns
    -------
    (x_cf, train_index, distance) or None if no valid NUN found
    """
    dist_fn = _dtw_distance if metric == "dtw" else _euclidean_distance

    target_mask = y_train == c_target
    target_indices = np.where(target_mask)[0]

    if len(target_indices) == 0:
        return None

    candidates = []
    for idx in target_indices:
        candidate = X_train[idx]

        # Confidence filter
        if min_confidence > 0 and hasattr(model, "predict_proba"):
            proba = model.predict_proba(candidate[np.newaxis])[0]
            if proba[c_target] < min_confidence:
                continue

        d = dist_fn(x, candidate)
        candidates.append((candidate, idx, d))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[2])
    return candidates[0]

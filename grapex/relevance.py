"""Gradual-pattern relevance (Section III of the paper).

Axis conventions used throughout this module:
  x              : np.ndarray of shape (C, T) — C channels, T time steps
  gp.regions     : [(ts, te)]                 — indices along T
  item.attribute : channel index along C
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .msgp import GradualPattern
from .nun import find_nun


def cov(gp: GradualPattern, x: np.ndarray) -> bool:
    """Definition 1 — x satisfies gp iff every item follows its direction on [ts, te]."""
    ts, te = gp.regions[0]
    for item in gp.items:
        c, d = item.attribute, item.variation
        if d == "↑" and not (x[c, ts] < x[c, te]):
            return False
        if d == "↓" and not (x[c, ts] > x[c, te]):
            return False
    return True


def cov_soft(gp: GradualPattern, x: np.ndarray) -> float:
    """Fraction of the items of gp satisfied by x (continuous version of Definition 1)."""
    if not gp.items:
        return 0.0
    ts, te = gp.regions[0]
    ok = sum(
        1 for item in gp.items
        if (item.variation == "↑") == (x[item.attribute, ts] < x[item.attribute, te])
    )
    return ok / len(gp.items)


def substitute(x: np.ndarray, x_cf: np.ndarray, gp: GradualPattern) -> np.ndarray:
    """Definition 2 — x ⊕_gp x_cf: replace the pattern channels on [ts, te] by x_cf."""
    ts, te = gp.regions[0]
    x_mod = x.copy()
    for item in gp.items:
        x_mod[item.attribute, ts:te + 1] = x_cf[item.attribute, ts:te + 1]
    return x_mod


def delta_f(gp: GradualPattern, x: np.ndarray, x_cf: np.ndarray,
            model, c_target: int) -> float:
    """Eq. (3) — Δf(gp, x, x_cf) = f_c(x ⊕_gp x_cf) − f_c(x)."""
    x_mod = substitute(x, x_cf, gp)
    prob_orig = model.predict_proba(x[np.newaxis])[0][c_target]
    prob_mod = model.predict_proba(x_mod[np.newaxis])[0][c_target]
    return float(prob_mod - prob_orig)


def is_gp_relevant(gp: GradualPattern, x: np.ndarray, x_cf: np.ndarray,
                   model, c_target: int, tau: float = 0.01) -> Dict[str, Any]:
    """Relevance conditions C1 ∧ C2 ∧ C3.

    Returns a dict with c1, c2, c3, delta, relevant, cov_x, cov_cf.
    delta is always computed (useful for diagnostics even when C1/C2 fail).
    """
    c1 = not cov(gp, x)        # C1: the query does NOT satisfy gp
    c2 = cov(gp, x_cf)         # C2: the target representative satisfies gp
    dv = delta_f(gp, x, x_cf, model, c_target)
    c3 = dv > tau              # C3: substitution increases target-class confidence

    return {
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "delta": dv,
        "relevant": c1 and c2 and c3,
        "cov_x": cov_soft(gp, x),
        "cov_cf": cov_soft(gp, x_cf),
    }


def relevance_score(gp: GradualPattern, delta: float, T: int) -> float:
    """Eq. (4) — score = Δf · support · (length / T)."""
    return delta * gp.support * (gp.length / T)


def relevance_counterfactual(x: np.ndarray, target_class: int,
                             X_train: np.ndarray, y_train: np.ndarray, model,
                             patterns: List[GradualPattern],
                             k: int = 3, tau: float = 0.01) -> Optional[np.ndarray]:
    """Counterfactual built from the top-k relevant target-class patterns.

      1. Find the NUN of x in the training set for target_class.
      2. Test every target-class pattern with is_gp_relevant().
      3. Keep the top-k relevant patterns (fallback: top-k by support, then length).
      4. Substitute each selected pattern's channels/window from the NUN.

    Returns the counterfactual as an array of shape (C, T), or None.
    """
    nun_result = find_nun(x, X_train, y_train, model,
                          c_target=target_class, metric="euclidean",
                          min_confidence=0.0)
    if nun_result is None or not patterns:
        return None
    x_nun, _, _ = nun_result

    T = x.shape[1]
    scored = []
    for gp_id, gp in enumerate(patterns):
        r = is_gp_relevant(gp, x, x_nun, model, target_class, tau=tau)
        scored.append((relevance_score(gp, r["delta"], T), gp_id, gp, r["relevant"]))

    relevant = [(s, gid, gp) for s, gid, gp, rel in scored if rel]
    if relevant:
        selected = [gp for _, _, gp in sorted(relevant, key=lambda t: t[:2], reverse=True)[:k]]
    else:
        selected = sorted(patterns, key=lambda p: (-p.support, -p.length))[:k]

    cf = x.copy()
    for gp in selected:
        cf = substitute(cf, x_nun, gp)
    return cf

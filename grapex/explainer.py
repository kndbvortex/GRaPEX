import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import euclidean
from tslearn.metrics import dtw

from .msgp import MSGPMax, GradualPattern


class GraPEX:
    """GraPEX counterfactual explainer (pattern extraction, NUN search, interpolation)."""

    def __init__(self):
        self.class_patterns: Dict[int, List[GradualPattern]] = {}
        self.class_timestamps: Dict[int, Dict[str, List[Tuple[int, int]]]] = {}
        self.reference_cf = None

    def extract_class_patterns(self, X_train: np.ndarray, y_train: np.ndarray,
                               min_support: float = 0.5, min_length: int = 7):
        unique_classes = np.unique(y_train)

        for cls in unique_classes:
            class_indices = np.where(y_train == cls)[0]
            class_samples = X_train[class_indices]
            data = np.transpose(class_samples, (0, 2, 1))

            print(f"\n=== Class {cls} ===")
            print(f"Shape: {data.shape} (samples × timesteps × channels)")

            n_channels = data.shape[2]
            col_names = [f"var_{i}" for i in range(n_channels)]

            msgp = MSGPMax(data, column_names=col_names)
            patterns = msgp.extract_patterns(min_support=min_support, min_length=min_length)

            self.class_patterns[cls] = patterns
            self.class_timestamps[cls] = self._extract_timestamps(patterns)

            print(f"Found {len(patterns)} patterns")
            for p in patterns[:5]:
                print(f"  {p}")

    def _extract_timestamps(self, patterns: List[GradualPattern]) -> Dict[str, List[Tuple[int, int]]]:
        timestamps = {}
        for pattern in patterns:
            key = " ".join(str(item) for item in sorted(pattern.items, key=lambda x: x.attribute))
            timestamps[key] = pattern.regions
        return timestamps

    def find_counterfactual_templates(self, x: np.ndarray, X_train: np.ndarray, y_train: np.ndarray,
                                      target_class: int, model, k: int = 5, metric: str = "dtw") -> List[Tuple[np.ndarray, int, float]]:
        """Find k nearest templates from target class that are correctly classified."""
        target_indices = np.where(y_train == target_class)[0]
        X_target = X_train[target_indices]

        candidates = []
        for idx, x_candidate in enumerate(X_target):
            pred = model.predict(x_candidate.reshape(1, x_candidate.shape[0], x_candidate.shape[1]))[0]
            if pred == target_class:
                if metric == "euclidean":
                    dist = np.mean([euclidean(x[i], x_candidate[i]) for i in range(x.shape[0])])
                else:
                    dist = np.mean([dtw(x[i], x_candidate[i]) for i in range(x.shape[0])])
                candidates.append((x_candidate, target_indices[idx], dist))

        candidates.sort(key=lambda t: t[2])
        return candidates[:k]

    def _compute_change_mask(self, x: np.ndarray, x_cf: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Compute binary mask of significant changes."""
        diff = np.abs(x - x_cf)
        return (diff > threshold * np.abs(x).max()).astype(float)

    def _apply_pattern_modification(self, x: np.ndarray, x_template: np.ndarray,
                                    regions: List[Tuple[int, int]], alpha: float) -> np.ndarray:
        """Apply modification only in specified regions."""
        x_mod = x.copy()
        for start, end in regions:
            start = max(0, min(start, x.shape[1] - 1))
            end = max(start + 1, min(end + 1, x.shape[1]))
            x_mod[:, start:end] = (1 - alpha) * x[:, start:end] + alpha * x_template[:, start:end]
        return x_mod

    def _get_discriminative_regions(self, x: np.ndarray, x_template: np.ndarray,
                                    model, target_class: int, n_segments: int = 10) -> List[Tuple[int, int, float]]:
        """Find regions that most affect classification using ablation."""
        seq_len = x.shape[1]
        segment_len = seq_len // n_segments
        region_importance = []

        for i in range(n_segments):
            start = i * segment_len
            end = min((i + 1) * segment_len, seq_len)

            x_test = x.copy()
            x_test[:, start:end] = x_template[:, start:end]

            pred = model.predict(x_test.reshape(1, x_test.shape[0], x_test.shape[1]))[0]
            importance = 1.0 if pred == target_class else 0.0

            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(x_test.reshape(1, x_test.shape[0], x_test.shape[1]))[0]
                    importance = proba[target_class]
                except:
                    pass

            region_importance.append((start, end, importance))

        region_importance.sort(key=lambda r: r[2], reverse=True)
        return region_importance

    def generate_counterfactual_progressive(self, x: np.ndarray, x_template: np.ndarray,
                                           target_class: int, model,
                                           pattern_regions: List[Tuple[int, int]],
                                           max_iterations: int = 20) -> Tuple[np.ndarray, List[Tuple], float, str]:
        """Generate counterfactual using progressive modification strategy."""

        # Strategy 1: Pattern-based with increasing alpha
        if pattern_regions:
            for alpha in [0.3, 0.5, 0.7, 0.9, 1.0]:
                x_cf = self._apply_pattern_modification(x, x_template, pattern_regions, alpha)
                pred = model.predict(x_cf.reshape(1, x_cf.shape[0], x_cf.shape[1]))[0]
                if pred == target_class:
                    return x_cf, pattern_regions, alpha, "pattern_based"

        # Strategy 2: Discriminative region detection
        disc_regions = self._get_discriminative_regions(x, x_template, model, target_class)
        cumulative_regions = []

        for start, end, _ in disc_regions:
            cumulative_regions.append((start, end))
            for alpha in [0.5, 0.7, 0.9, 1.0]:
                x_cf = self._apply_pattern_modification(x, x_template, cumulative_regions, alpha)
                pred = model.predict(x_cf.reshape(1, x_cf.shape[0], x_cf.shape[1]))[0]
                if pred == target_class:
                    return x_cf, cumulative_regions, alpha, "discriminative"

        # Strategy 3: Binary search on global interpolation
        low, high = 0.0, 1.0
        best_cf = None
        best_alpha = 1.0

        for _ in range(max_iterations):
            mid = (low + high) / 2
            x_cf = (1 - mid) * x + mid * x_template
            pred = model.predict(x_cf.reshape(1, x_cf.shape[0], x_cf.shape[1]))[0]

            if pred == target_class:
                best_cf = x_cf.copy()
                best_alpha = mid
                high = mid
            else:
                low = mid

            if high - low < 0.01:
                break

        if best_cf is not None:
            return best_cf, [(0, x.shape[1])], best_alpha, "binary_search"

        # Strategy 4: Full template as last resort
        pred = model.predict(x_template.reshape(1, x_template.shape[0], x_template.shape[1]))[0]
        if pred == target_class:
            return x_template.copy(), [(0, x.shape[1])], 1.0, "full_template"

        # Fallback: return best effort
        return (1 - 0.7) * x + 0.7 * x_template, [(0, x.shape[1])], 0.7, "fallback"

    def explain(self, x: np.ndarray, X_train: np.ndarray, y_train: np.ndarray,
                predicted_class: int, target_class: int = None, metric: str = "dtw",
                alpha: float = 0.5, model=None, optimize: bool = True) -> Dict:

        if target_class is None:
            unique_classes = np.unique(y_train)
            other_classes = [c for c in unique_classes if c != predicted_class]
            target_class = other_classes[0] if other_classes else predicted_class

        # Get pattern regions for target class
        target_timestamps = self.class_timestamps.get(target_class, {})
        pattern_regions = []
        for regions in target_timestamps.values():
            pattern_regions.extend(regions)
        pattern_regions = list(set(pattern_regions))

        # If no patterns found, return original instance
        if not pattern_regions and not self.class_patterns.get(target_class):
            print(f"   No patterns found for target class {target_class} - returning original instance")
            return {
                "original": x,
                "counterfactual_template": x,
                "counterfactual": x.copy(),
                "important_regions": [],
                "source_class": predicted_class,
                "target_class": target_class,
                "cf_train_index": None,
                "target_patterns": {},
                "alpha_used": 0.0,
                "method_used": "no_patterns",
                "valid": False,
                "sparsity": 0.0,
                "no_patterns": True,
                "metric": metric
            }

        # Find multiple valid templates
        templates = self.find_counterfactual_templates(x, X_train, y_train, target_class, model, k=5, metric=metric)

        if not templates:
            # No valid templates found - return original
            print(f"   No valid templates found for target class {target_class} - returning original instance")
            return {
                "original": x,
                "counterfactual_template": x,
                "counterfactual": x.copy(),
                "important_regions": [],
                "source_class": predicted_class,
                "target_class": target_class,
                "cf_train_index": None,
                "target_patterns": target_timestamps,
                "alpha_used": 0.0,
                "method_used": "no_templates",
                "valid": False,
                "sparsity": 0.0,
                "no_patterns": True,
                "metric": metric
            }

        best_result = None
        best_sparsity = float('inf')

        for x_template, cf_idx, dist in templates:
            if optimize and model is not None:
                x_cf, regions, alpha_used, method = self.generate_counterfactual_progressive(
                    x, x_template, target_class, model, pattern_regions
                )

                pred = model.predict(x_cf.reshape(1, x_cf.shape[0], x_cf.shape[1]))[0]
                is_valid = (pred == target_class)

                # Calculate sparsity (prefer sparser solutions)
                sparsity = np.sum(np.abs(x - x_cf) > 0.01 * np.abs(x).max()) / x.size

                if is_valid and sparsity < best_sparsity:
                    best_sparsity = sparsity
                    best_result = {
                        "original": x,
                        "counterfactual_template": x_template,
                        "counterfactual": x_cf,
                        "important_regions": regions,
                        "source_class": predicted_class,
                        "target_class": target_class,
                        "cf_train_index": cf_idx,
                        "target_patterns": target_timestamps,
                        "alpha_used": alpha_used,
                        "method_used": method,
                        "valid": True,
                        "sparsity": sparsity,
                        "no_patterns": False,
                        "metric": metric
                    }

        if best_result is not None:
            return best_result

        # No valid counterfactual found - return original instance
        print(f"   No valid counterfactual found - returning original instance")
        return {
            "original": x,
            "counterfactual_template": x,
            "counterfactual": x.copy(),
            "important_regions": [],
            "source_class": predicted_class,
            "target_class": target_class,
            "cf_train_index": None,
            "target_patterns": target_timestamps,
            "alpha_used": 0.0,
            "method_used": "no_valid_cf",
            "valid": False,
            "sparsity": 0.0,
            "no_patterns": True,
            "metric": metric
        }

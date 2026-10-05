"""
model_compat.py — Format-agnostic Keras model loader.

Models in this repo exist in two formats:
  - tf_keras format  (saved with tf_keras 2.x): module = 'tf_keras.src.engine.functional'
  - keras 3 format   (saved with keras 3.x):    module = 'keras.src.models.functional'

This module detects the format from the saved config.json and dispatches
to the correct loader.
"""

import zipfile
import json
from pathlib import Path


def _detect_format(model_path: str) -> str:
    """Return 'tf_keras' or 'keras3' based on the config.json inside the archive."""
    with zipfile.ZipFile(model_path) as z:
        cfg = json.load(z.open("config.json"))
    module = cfg.get("module", "")
    return "tf_keras" if "tf_keras" in module else "keras3"


def load_model_raw(model_path):
    """Load a model with the correct backend; return the raw Keras model object."""
    fmt = _detect_format(str(model_path))
    if fmt == "tf_keras":
        import os; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tf_keras
        return tf_keras.models.load_model(str(model_path))
    else:
        import os; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import keras
        return keras.models.load_model(str(model_path))

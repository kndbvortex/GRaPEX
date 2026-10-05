from . import Saliency_Base, SaliencyMethods_PTY

# Lazy import for TensorFlow methods to avoid Keras 3 compatibility issues
def __getattr__(name):
    if name == "SaliencyMethods_TF":
        from . import SaliencyMethods_TF
        return SaliencyMethods_TF
    if name == "TSR":
        from . import TSR
        return TSR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Saliency_Base", "SaliencyMethods_PTY", "SaliencyMethods_TF", "TSR"]

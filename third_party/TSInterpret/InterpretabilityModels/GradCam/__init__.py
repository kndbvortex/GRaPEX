def __getattr__(name):
    if name == "GradCam_1D":
        from . import GradCam_1D
        return GradCam_1D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["GradCam_1D"]

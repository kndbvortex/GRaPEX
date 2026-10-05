from . import CF, COMTE, TSEvo, TSEvoCF, COMTECF, SETSCF

def __getattr__(name):
    if name == "NativeGuideCF":
        from . import NativeGuideCF
        return NativeGuideCF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["CF", "COMTE", "NativeGuideCF", "TSEvoCF", "TSEvo", "COMTECF", "SETSCF"]

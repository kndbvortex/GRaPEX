import torch

from TSInterpret.InterpretabilityModels.Saliency.SaliencyMethods_PTY import Saliency_PTY


class TSR:
    """Wrapper Class for Saliency Calculation. Automatically calls the corresponding PYT or TF implementation."""

    def __new__(
        self, model, NumTimeSteps, NumFeatures, method="GRAD", mode="time", device="cpu", normalize=True, tsr=True
    ):
        if isinstance(model, torch.nn.Module):
            return Saliency_PTY(
                model,
                NumTimeSteps,
                NumFeatures,
                method=method,
                mode=mode,
                device=device,
                normalize=normalize,
                tsr=tsr
            )
        else:
            try:
                import tensorflow
                if isinstance(model, tensorflow.keras.Model):
                    from TSInterpret.InterpretabilityModels.Saliency.SaliencyMethods_TF import Saliency_TF
                    return Saliency_TF(
                        model, NumTimeSteps, NumFeatures, method=method, mode=mode, tsr=tsr
                    )
            except (ImportError, AttributeError):
                pass

            raise NotImplementedError(
                "Please use a TF or PYT Classification model! "
                "If the current model is a TF or PYT Model, "
                "try calling the wrappers directly "
                "(TF -> Saliency_TF, PYT -> Saliency_PYT)"
            )

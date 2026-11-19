"""Gaussian mixture cluster-count inference API for the unchanged test_models.py.

Include these files beside this module when packaging the project:
  model_assets/gaussian/symmetric.pt     trained Set Transformer checkpoint
  model_assets/gaussian/baseline.pt      trained MLP checkpoint
  model_assets/gaussian/sample_batch.pt  held-out test point sets (B, 100, 2)

Use the Gaussian checkpoints produced by train.py:
gaussian_set_fraction_100.pt and gaussian_baseline_fraction_100.pt.
The four output scores represent one, two, three, or four clusters.

Checkpoints may contain a state_dict directly or train.py's checkpoint dictionary.
Samples may contain an input tensor directly or a dictionary with an 'inputs'
tensor (and optional labels, which sample_batch does not return).
"""

from pathlib import Path

import torch

from baseline import MLPBaseline
from set_transformer import SetTransformer


_ASSETS = Path(__file__).resolve().parent / "model_assets" / "gaussian"


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_asset(name):
    path = _ASSETS / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing packaged artifact: {path}. Include the trained checkpoint "
            "or real test samples before running test_models.py."
        )
    return torch.load(path, map_location="cpu", weights_only=True)


class _SetClassifier(SetTransformer):
    def forward(self, inputs):
        return super().forward(inputs).squeeze(1)


def load_model(kind):   # kind is "symmetric" or "baseline"
    """Load the trained weights installed with the package onto the GPU if one is available, otherwise the CPU, and return the model."""
    if kind not in ("symmetric", "baseline"):
        raise ValueError("kind must be 'symmetric' or 'baseline'")
    checkpoint = _load_asset(f"{kind}.pt")
    state = checkpoint.get("state_dict", checkpoint)
    model = (_SetClassifier(dim_input=2, dim_output=4)
             if kind == "symmetric" else
             MLPBaseline(set_size=100, dim_input=2, dim_output=4))
    model.load_state_dict(state, strict=True)
    return model.to(_device()).eval()


def sample_batch():
    """Return a few test examples installed with the package, on the same device as the model."""
    saved = _load_asset("sample_batch.pt")
    inputs = saved["inputs"] if isinstance(saved, dict) else saved
    if (not isinstance(inputs, torch.Tensor) or inputs.ndim != 3
            or inputs.shape[0] == 0 or inputs.shape[1:] != (100, 2)):
        raise ValueError("Gaussian test inputs must have shape (B, 100, 2)")
    if not inputs.is_floating_point() or not torch.isfinite(inputs).all():
        raise ValueError("Test inputs must contain finite floating-point features")
    return inputs.to(_device(), dtype=torch.float32)

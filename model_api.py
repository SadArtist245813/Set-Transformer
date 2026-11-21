
"""Pretrained CIFAR-10 inference API."""

from pathlib import Path

import torch

from baseline import MLPBaseline
from set_transformer import SetTransformer


_ASSETS = Path(__file__).resolve().parent / "model_assets"


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_asset(name):
    path = _ASSETS / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing packaged artifact: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


class _SetClassifier(SetTransformer):
    def forward(self, inputs):
        return super().forward(inputs).squeeze(1)


def load_model(kind):
    if kind not in ("symmetric", "baseline"):
        raise ValueError("kind must be 'symmetric' or 'baseline'")

    checkpoint = _load_asset(f"{kind}.pt")
    state = checkpoint.get("state_dict", checkpoint)

    if kind == "symmetric":
        model = _SetClassifier(dim_input=194, dim_output=10)
    else:
        model = MLPBaseline(
            set_size=144,
            dim_input=194,
            dim_output=10
        )

    model.load_state_dict(state, strict=True)
    return model.to(_device()).eval()


def sample_batch():
    saved = _load_asset("sample_batch.pt")
    inputs = saved["inputs"] if isinstance(saved, dict) else saved

    if (
        not isinstance(inputs, torch.Tensor)
        or inputs.ndim != 3
        or inputs.shape[0] == 0
        or inputs.shape[1:] != (144, 194)
    ):
        raise ValueError("Expected CIFAR-10 inputs shaped (B, 144, 194)")

    return inputs.to(_device(), dtype=torch.float32)

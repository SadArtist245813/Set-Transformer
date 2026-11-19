import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    Flattened MLP baseline.

    Unlike the Set Transformer, this model treats the input
    as a fixed-order vector, so shuffling the set can
    change its prediction.
    """

    def __init__(
        self,
        set_size=100,
        dim_input=1,
        dim_hidden=420,
        dim_output=1
    ):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=1)

        self.network = nn.Sequential(
            nn.Linear(set_size * dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, X):
        X = self.flatten(X)
        return self.network(X)


if __name__ == "__main__":
    model = MLPBaseline()

    X = torch.rand(8, 100, 1)
    output = model(X)

    print("Input shape:", X.shape)
    print("Output shape:", output.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

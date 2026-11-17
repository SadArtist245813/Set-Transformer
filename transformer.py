import torch
import torch.nn as nn

class StandardTransformer(nn.Module):
    def __init__(self, dim_input=1, dim_hidden=128,
                 num_heads=4, num_layers=2, dim_output=1):
        super().__init__()

        self.embedding = nn.Linear(dim_input, dim_hidden)

        layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden,
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

        self.output = nn.Linear(dim_hidden, dim_output)

    def forward(self, X):
        X = self.embedding(X)
        X = self.encoder(X)
        X = X.mean(dim=1, keepdim=True)
        return self.output(X)

if __name__ == "__main__":
    X = torch.rand(8, 100, 1)

    model = StandardTransformer()
    output = model(X)

    print("Standard Transformer:", output.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
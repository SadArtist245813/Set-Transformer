import torch
import torch.nn as nn
import torch.nn.functional as F


def split_heads(x, n_head):
    """(B, T, C) -> (B, n_head, T, C // n_head)"""
    return x.unflatten(-1, (n_head, -1)).transpose(1, 2)


def merge_heads(x):
    """(B, n_head, T, D) -> (B, T, n_head * D)"""
    return x.transpose(1, 2).flatten(2)


class MAB_KV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_q = nn.Linear(128, 128)
        self.fc_k = nn.Linear(1, 128)
        self.fc_v = nn.Linear(1, 128)
        self.fc_o = nn.Linear(128, 128)

    def forward(self, query, context):
        h = split_heads(self.fc_q(query), 4)  # (e, 4, tq, 128/4)
        x = h @ split_heads(self.fc_k(context), 4).transpose(-2, -1) / 11.313708498984761  # (e, 4, tq, tk)
        x = torch.softmax(x, dim=-1) @ split_heads(self.fc_v(context), 4)  # (e, 4, tq, 128/4)
        y = merge_heads(h + x)  # (e, tq, 128)
        return y + F.relu(self.fc_o(y))


class MAB_Q2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_q = nn.Linear(1, 128)
        self.fc_k = nn.Linear(128, 128)
        self.fc_v = nn.Linear(128, 128)
        self.fc_o = nn.Linear(128, 128)

    def forward(self, query, context):
        h = split_heads(self.fc_q(query), 4)  # (e, 4, tq, 128/4)
        x = h @ split_heads(self.fc_k(context), 4).transpose(-2, -1) / 11.313708498984761  # (e, 4, tq, tk)
        x = torch.softmax(x, dim=-1) @ split_heads(self.fc_v(context), 4)  # (e, 4, tq, 128/4)
        y = merge_heads(h + x)  # (e, tq, 128)
        return y + F.relu(self.fc_o(y))


class ISAB_Input(nn.Module):
    def __init__(self):
        super().__init__()
        self.I = nn.Parameter(torch.zeros(1, 32, 128))
        nn.init.normal_(self.I, std=0.02)
        self.mab0 = MAB_KV2()
        self.mab1 = MAB_Q2()

    def forward(self, X):
        x = self.mab0(self.I, X)  # (e, 32, 128)
        return self.mab1(X, x)


class MAB128(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_q = nn.Linear(128, 128)
        self.fc_k = nn.Linear(128, 128)
        self.fc_v = nn.Linear(128, 128)
        self.fc_o = nn.Linear(128, 128)

    def forward(self, query, context):
        h = split_heads(self.fc_q(query), 4)  # (e, 4, tq, 128/4)
        x = h @ split_heads(self.fc_k(context), 4).transpose(-2, -1) / 11.313708498984761  # (e, 4, tq, tk)
        x = torch.softmax(x, dim=-1) @ split_heads(self.fc_v(context), 4)  # (e, 4, tq, 128/4)
        y = merge_heads(h + x)  # (e, tq, 128)
        return y + F.relu(self.fc_o(y))


class ISAB_Hidden(nn.Module):
    def __init__(self):
        super().__init__()
        self.I = nn.Parameter(torch.zeros(1, 32, 128))
        nn.init.normal_(self.I, std=0.02)
        self.mab0 = MAB128()
        self.mab1 = MAB128()

    def forward(self, X):
        x = self.mab0(self.I, X)  # (e, 32, 128)
        return self.mab1(X, x)


class Pooling_by_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.S = nn.Parameter(torch.zeros(1, 1, 128))
        nn.init.normal_(self.S, std=0.02)
        self.mab = MAB128()

    def forward(self, X):
        return self.mab(self.S, X)


class Set_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.mab = MAB128()

    def forward(self, X):
        return self.mab(X, X)


class Set_Transformer___Assignment(nn.Module):
    def __init__(self):
        super().__init__()
        self.isab1 = ISAB_Input()
        self.isab2 = ISAB_Hidden()
        self.pma = Pooling_by_Attention()
        self.sab1 = Set_Attention()
        self.sab2 = Set_Attention()
        self.regression = nn.Linear(128, 1)

    def forward(self, X):
        x = self.isab1(X)  # (e, 10, 128)
        x = self.isab2(x)  # (e, 10, 128)
        x = self.pma(x)  # (e, 1, 128)
        x = self.sab1(x)  # (e, 1, 128)
        x = self.sab2(x)  # (e, 1, 128)
        return self.regression(x)

    def loss(self, predicted_max, y):
        """Mean squared error between predicted_max and y."""
        return F.mse_loss(predicted_max, y)

    @torch.no_grad()
    def predict(self, X):
        """Inference: forward() as is (the loss works on the raw output)."""
        return self(X)


if __name__ == "__main__":
    model = Set_Transformer___Assignment()
    print(sum(p.numel() for p in model.parameters()), "parameters")
    X = torch.randn(8, 10, 1)
    predicted_max = model(X)
    print(predicted_max.shape)  # expected torch.Size([8, 1, 1])
    y = torch.randn(8, 1, 1)
    print(model.loss(predicted_max, y).item(), "loss")

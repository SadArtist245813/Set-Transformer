import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
def causal_mask(scores):
    """Hide future positions: scores above the diagonal become -inf."""
    t = scores.size(-1)
    future = torch.ones(t, t, dtype=torch.bool, device=scores.device).triu(1)
    return scores.masked_fill(future, float("-inf"))
 
 
def sinusoidal_pe(x, base=10000.0):
    """The fixed sine/cosine position encoding of "Attention Is All You Need" for x (B, T, D), as a (T, D) table to add
    to x: feature 2i of position t is sin(t / base^(2i/D)), feature 2i + 1 the cosine. Only x's sizes are used."""
    t, d = x.size(1), x.size(2)
    pos = torch.arange(t, device=x.device, dtype=torch.float32)[:, None]
    inv = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = pos * inv
    return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1).flatten(-2).to(x.dtype)
 
 
class EncoderLayer(nn.Module):
    def __init__(self, d=512, n_head=8, ff=2048):
        super().__init__()
        self.n_head = n_head
        self.d = d
        self.sa_q = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.sa_q.weight, gain=0.70710678)
        nn.init.zeros_(self.sa_q.bias)
        self.sa_k = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.sa_k.weight, gain=0.70710678)
        nn.init.zeros_(self.sa_k.bias)
        self.sa_drop = nn.Dropout(0.1)
        self.sa_v = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.sa_v.weight, gain=0.70710678)
        nn.init.zeros_(self.sa_v.bias)
        self.sa_out = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.sa_out.weight)
        nn.init.zeros_(self.sa_out.bias)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(d, eps=1e-5)
        self.linear1 = nn.Linear(d, ff)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(ff, d)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(d, eps=1e-5)
 
    def forward(self, x):
        q = self.sa_q(x).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, t, d/n_head)
        q = q * (self.d / self.n_head) ** -0.5  # (e, n_head, t, d/n_head)
        k = self.sa_k(x).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, t, d/n_head)
        att = q @ k.transpose(-2, -1)  # (e, n_head, t, t)
        att = torch.softmax(att, dim=-1)  # (e, n_head, t, t)
        att = self.sa_drop(att)  # (e, n_head, t, t)
        v = self.sa_v(x).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, t, d/n_head)
        y = att @ v  # (e, n_head, t, d/n_head)
        y = y.transpose(1, 2).flatten(2)  # (e, t, d)
        h = self.sa_out(y)  # (e, t, d)
        x = x + self.dropout1(h)  # (e, t, d)
        x = self.norm1(x)  # (e, t, d)
        a = F.relu(self.linear1(x))  # (e, t, ff)
        h = self.dropout(a)  # (e, t, ff)
        h = self.linear2(h)  # (e, t, d)
        x = x + self.dropout2(h)  # (e, t, d)
        return self.norm2(x)
 
 
class DecoderLayer(nn.Module):
    def __init__(self, d=512, n_head=8, ff=2048):
        super().__init__()
        self.n_head = n_head
        self.d = d
        self.sa_q = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.sa_q.weight, gain=0.70710678)
        nn.init.zeros_(self.sa_q.bias)
        self.sa_k = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.sa_k.weight, gain=0.70710678)
        nn.init.zeros_(self.sa_k.bias)
        self.sa_drop = nn.Dropout(0.1)
        self.sa_v = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.sa_v.weight, gain=0.70710678)
        nn.init.zeros_(self.sa_v.bias)
        self.sa_out = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.sa_out.weight)
        nn.init.zeros_(self.sa_out.bias)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(d, eps=1e-5)
        self.ca_q = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.ca_q.weight, gain=0.70710678)
        nn.init.zeros_(self.ca_q.bias)
        self.ca_k = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.ca_k.weight, gain=0.70710678)
        nn.init.zeros_(self.ca_k.bias)
        self.ca_drop = nn.Dropout(0.1)
        self.ca_v = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.ca_v.weight, gain=0.70710678)
        nn.init.zeros_(self.ca_v.bias)
        self.ca_out = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.ca_out.weight)
        nn.init.zeros_(self.ca_out.bias)
        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(d, eps=1e-5)
        self.linear1 = nn.Linear(d, ff)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(ff, d)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.dropout3 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(d, eps=1e-5)
 
    def forward(self, x, memory):
        q = self.sa_q(x).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, t, d/n_head)
        q = q * (self.d / self.n_head) ** -0.5  # (e, n_head, t, d/n_head)
        k = self.sa_k(x).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, t, d/n_head)
        att = q @ k.transpose(-2, -1)  # (e, n_head, t, t)
        att = causal_mask(att)  # (e, n_head, t, t)
        att = torch.softmax(att, dim=-1)  # (e, n_head, t, t)
        att = self.sa_drop(att)  # (e, n_head, t, t)
        v = self.sa_v(x).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, t, d/n_head)
        y = att @ v  # (e, n_head, t, d/n_head)
        y = y.transpose(1, 2).flatten(2)  # (e, t, d)
        h = self.sa_out(y)  # (e, t, d)
        x = x + self.dropout1(h)  # (e, t, d)
        x = self.norm1(x)  # (e, t, d)
        q = self.ca_q(x).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, t, d/n_head)
        q = q * (self.d / self.n_head) ** -0.5  # (e, n_head, t, d/n_head)
        k = self.ca_k(memory).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, s, d/n_head)
        att = q @ k.transpose(-2, -1)  # (e, n_head, t, s)
        att = torch.softmax(att, dim=-1)  # (e, n_head, t, s)
        att = self.ca_drop(att)  # (e, n_head, t, s)
        v = self.ca_v(memory).unflatten(-1, (self.n_head, -1)).transpose(1, 2)  # (e, n_head, s, d/n_head)
        y = att @ v  # (e, n_head, t, d/n_head)
        y = y.transpose(1, 2).flatten(2)  # (e, t, d)
        h = self.ca_out(y)  # (e, t, d)
        x = x + self.dropout2(h)  # (e, t, d)
        x = self.norm2(x)  # (e, t, d)
        a = F.relu(self.linear1(x))  # (e, t, ff)
        h = self.dropout(a)  # (e, t, ff)
        h = self.linear2(h)  # (e, t, d)
        x = x + self.dropout3(h)  # (e, t, d)
        return self.norm3(x)
 
 
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tgt_embedding = nn.Embedding(37000, 512)
        self.tgt_embedding_drop = nn.Dropout(0.1)
        self.embedding = nn.Embedding(37000, 512)
        nn.init.normal_(self.embedding.weight, std=0.04419417)
        self.embedding_drop = nn.Dropout(0.1)
        self.encoder = nn.ModuleList([EncoderLayer(512, 8, 2048) for _ in range(6)])
        self.encoder_norm = nn.LayerNorm(512, eps=1e-5)
        self.decoder = nn.ModuleList([DecoderLayer(512, 8, 2048) for _ in range(6)])
        self.decoder_norm = nn.LayerNorm(512, eps=1e-5)
        self.generator = nn.Linear(512, 37000, bias=False)
        self.tgt_embedding.weight = self.embedding.weight
        self.generator.weight = self.embedding.weight
 
    def forward(self, src, tgt):
        g = self.tgt_embedding(tgt) * 512 ** 0.5  # (e, 48, 512)
        x = g + sinusoidal_pe(g)  # (e, 48, 512)
        h = self.tgt_embedding_drop(x)  # (e, 48, 512)
        g = self.embedding(src) * 512 ** 0.5  # (e, 64, 512)
        x = g + sinusoidal_pe(g)  # (e, 64, 512)
        h2 = self.embedding_drop(x)  # (e, 64, 512)
        x = h2
        for blk in self.encoder:  # (e, 64, 512)
            x = blk(x)
        memory = self.encoder_norm(x)  # (e, 64, 512)
        x = h
        for blk in self.decoder:  # (e, 48, 512)
            x = blk(x, memory)
        n = self.decoder_norm(x)  # (e, 48, 512)
        return self.generator(n)
 
    def loss(self, logits, tgt):
        """Next-token cross-entropy: position t of logits is scored on token t + 1 of tgt."""
        return F.cross_entropy(logits[:, :-1].flatten(0, -2), tgt[:, 1:].flatten(), label_smoothing=0.1)
 
    @torch.no_grad()
    def predict(self, src, tgt):
        """Inference: forward() followed by softmax, which the loss leaves out."""
        return torch.softmax(self(src, tgt), dim=-1)
 
 
if __name__ == "__main__":
    model = Transformer()
    print(sum(p.numel() for p in model.parameters()), "parameters")
    src = torch.randint(0, 37000, (8, 64))  # (e=8, s=64)
    tgt = torch.randint(0, 37000, (8, 48))  # (e=8, t=48)
    logits = model(src, tgt)
    print(logits.shape)  # expected torch.Size([8, 48, 37000])
    print(model.loss(logits, tgt).item(), "loss")
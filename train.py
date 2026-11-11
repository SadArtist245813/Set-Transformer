"""Train Transformer: written by Torch Canvas, with model.py next to it.
 
The settings below come from the export dialog; change them right here. As written, the script trains
on stand-in random data with the right shapes, so you can check everything runs, then swap
make_dataset() for your own data."""
import os
import time
 
import torch
import torch.nn as nn
# import load_data as data
from torch.utils.data import DataLoader, TensorDataset, random_split
 
from model import Transformer
 
 
# ── settings (from the export dialog) ──
BATCH_SIZE = 32          # examples per step
SEQ_LEN = 48             # tokens per example
NUM_SAMPLES = 2048       # size of the stand-in dataset
VAL_FRACTION = 0.1       # share held out for validation
 
LR = 3e-4                # peak learning rate
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 100
MIN_LR_RATIO = 0.1       # the learning rate decays to this share of LR
EPOCHS = 10
GRAD_CLIP = 1.0          # max gradient norm
SEED = 0
 
LOG_EVERY = 50           # steps
EVAL_EVERY = 200         # steps
SAVE_EVERY = 1000        # steps
OUT_DIR = "checkpoints"
 
 
def make_dataset():
    """Stand-in random data with the shapes Transformer expects. Replace it with your own Dataset whose
    items are (src, tgt):
        src: token ids in [0, 37000)
        tgt: token ids in [0, 37000)
    Next-token targets are the inputs themselves, shifted by one inside loss().
    """
    n = NUM_SAMPLES
    src = torch.randint(0, 37000, (n, SEQ_LEN))
    tgt = torch.randint(0, 37000, (n, SEQ_LEN))
    return TensorDataset(src, tgt)
 
 
def compute_loss(model, batch):
    """One batch: forward(), then loss() with the targets."""
    src, tgt = batch
    logits = model(src, tgt)
    return model.loss(logits, tgt)
 
 
def make_optimizer(model):
    """AdamW with weight decay on the weight matrices only: biases, norm weights and embeddings
    are left undecayed."""
    embeddings = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    decay = [p for p in model.parameters() if p.dim() >= 2 and id(p) not in embeddings]
    no_decay = [p for p in model.parameters() if (p.dim() < 2 or id(p) in embeddings)]
    groups = [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=LR, betas=(0.9, 0.95))
 
 
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    for batch in loader:
        batch = [b.to(device) for b in batch]
        total += compute_loss(model, batch).item()
    model.train()
    return total / len(loader)
 
 
def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
 
    model = Transformer().to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters on {device}")
 
    data = make_dataset()
    n_val = max(1, int(len(data) * VAL_FRACTION))
    train_set, val_set = random_split(data, [len(data) - n_val, n_val], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    total_steps = EPOCHS * len(train_loader)
 
    optimizer = make_optimizer(model)
    # linear warmup over WARMUP_STEPS, then cosine decay to MIN_LR_RATIO * LR
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / WARMUP_STEPS, total_iters=WARMUP_STEPS)
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - WARMUP_STEPS), eta_min=LR * MIN_LR_RATIO)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, decay], milestones=[WARMUP_STEPS])
 
    step = 0
 
    def log(metrics):
        print(f"step {step:>6}  " + "  ".join(f"{k} {v:.4g}" for k, v in metrics.items()))
 
    def save():
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "last.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step}, path)
        return path
 
    def validate():
        metrics = {"val/loss": evaluate(model, val_loader, device)}
        log(metrics)
 
    model.train()
    running, count, t0 = 0.0, 0, time.time()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            loss = compute_loss(model, batch)
            loss.backward()
            running += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            count += 1
 
            if step % LOG_EVERY == 0:
                log({"train/loss": running / count, "lr": optimizer.param_groups[0]["lr"], "grad_norm": grad_norm.item(), "sec/step": (time.time() - t0) / count})
                running, count, t0 = 0.0, 0, time.time()
            if step % EVAL_EVERY == 0 and step < total_steps:
                validate()
            if step % SAVE_EVERY == 0 and step < total_steps:
                save()
 
    validate()
    print("saved", save())
 
 
if __name__ == "__main__":
    main()
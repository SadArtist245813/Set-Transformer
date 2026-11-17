"""Train Transformer architectures and a flattened MLP baseline on four datasets.

Expected files:
  transformer.py: StandardTransformer
  set_transformer.py: SetTransformer, ISAB_PMA, SAB_PMA
  load_data.py: load_NumericalSets, load_ModelNet40, load_CIFAR100, load_Omniglot

Examples:
  python train.py --dataset numerical --epochs 1
  python train.py --dataset omniglot --epochs 1 --episodes 20 --eval-episodes 10
  python train.py --dataset cifar --model standard --epochs 10
"""

import argparse
import copy
import csv
import importlib
import inspect
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from transformer import StandardTransformer
from set_transformer import SetTransformer, ISAB_PMA, SAB_PMA
from baseline import MLPBaseline

SEED = 42
MODEL_CLASSES = {
    "standard": StandardTransformer,
    "set": SetTransformer,
    "isab": ISAB_PMA,
    "sab": SAB_PMA,
    "baseline": MLPBaseline,
}

SET_SIZES = {
    "numerical": 100,
    "modelnet": 1024,
    "cifar": 144,
    "omniglot": 144,
    "gaussian": 100,
}

CONFIG = {
    "numerical": (1, 1, "regression"),
    "modelnet": (3, 40, "classification"),
    "cifar": (194, 100, "classification"),
    "omniglot": (66, 64, "episodic"),
    "gaussian": (2, 4, "classification"),
}

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(name, dataset):
    input_dim, output_dim, _ = CONFIG[dataset]
    if name == "baseline":
        return MLPBaseline(
            set_size=SET_SIZES[dataset],
            dim_input=input_dim,
            dim_hidden=420,
            dim_output=output_dim,
        )
    cls = MODEL_CLASSES[name]
    params = inspect.signature(cls).parameters
    kwargs = {}
    if "dim_input" in params:
        kwargs["dim_input"] = input_dim
    elif "input_dim" in params:
        kwargs["input_dim"] = input_dim
    else:
        raise TypeError(f"{cls.__name__} must accept dim_input or input_dim")
    if "dim_output" in params:
        kwargs["dim_output"] = output_dim
    elif "output_dim" in params:
        kwargs["output_dim"] = output_dim
    else:
        raise TypeError(f"{cls.__name__} must accept dim_output or output_dim")
    if "num_outputs" in params:
        kwargs["num_outputs"] = 1
    return cls(**kwargs)


def image_patches(images, patch_size=8):
    """(B,C,96,96) -> (B,144,C*64+2), with spatial coordinates."""
    b, c, h, w = images.shape
    if h % patch_size or w % patch_size:
        raise ValueError("Image dimensions must be divisible by 8")
    patches = F.unfold(images, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
    rows, cols = h // patch_size, w // patch_size
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, rows, device=images.device, dtype=images.dtype),
        torch.linspace(-1, 1, cols, device=images.device, dtype=images.dtype),
        indexing="ij",
    )
    coords = torch.stack((yy, xx), dim=-1).reshape(1, rows * cols, 2)
    return torch.cat((patches, coords.expand(b, -1, -1)), dim=-1)


def prepare_batch(batch, dataset, device):
    x, y = batch
    x = x.to(device, dtype=torch.float32)
    if dataset == "cifar":
        x = image_patches(x)
    y = y.to(device)
    if dataset == "numerical":
        y = y.float().reshape(-1)
    else:
        y = y.long().reshape(-1)
    return x, y


def predictions_2d(model, x, output_dim):
    out = model(x)
    if out.numel() != x.size(0) * output_dim:
        raise ValueError(
            f"Expected {x.size(0)} x {output_dim} outputs, got {tuple(out.shape)}"
        )
    return out.reshape(x.size(0), output_dim)


def ordinary_loss(model, x, y, dataset):
    _, output_dim, task = CONFIG[dataset]
    pred = predictions_2d(model, x, output_dim)
    if task == "regression":
        pred = pred[:, 0]
        loss = F.mse_loss(pred, y)
        metric_sum = (pred.detach() - y).abs().sum().item()  # MAE numerator
    else:
        loss = F.cross_entropy(pred, y)
        metric_sum = (pred.argmax(1) == y).sum().item()
    return loss, metric_sum, y.numel()


def run_ordinary_epoch(model, loader, dataset, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_metric = total_count = 0
    with torch.enable_grad() if training else torch.no_grad():
        for batch in loader:
            x, y = prepare_batch(batch, dataset, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, metric, count = ordinary_loss(model, x, y, dataset)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * count
            total_metric += metric
            total_count += count
    return total_loss / total_count, total_metric / total_count


def omniglot_class_indices(dataset):
    """Map Omniglot's character IDs to image indices without loading images."""
    if not hasattr(dataset, "_flat_character_images"):
        raise RuntimeError("Omniglot loader has changed; cannot access character labels")
    groups = defaultdict(list)
    for index, (_, character_id) in enumerate(dataset._flat_character_images):
        groups[int(character_id)].append(index)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def make_episode(dataset, class_indices, ways, rng):
    """One support and one query image for each of `ways` classes."""
    chosen = rng.sample(list(class_indices), ways)
    support, query = [], []
    for label in chosen:
        support_idx, query_idx = rng.sample(class_indices[label], 2)
        support.append(dataset[support_idx][0])
        query.append(dataset[query_idx][0])
    return torch.stack(support + query)  # (2*ways,C,96,96)


def episodic_loss(model, images, ways, device):
    images = image_patches(images.to(device, dtype=torch.float32))
    embeddings = predictions_2d(model, images, 64)
    support = F.normalize(embeddings[:ways], dim=-1)
    query = F.normalize(embeddings[ways:], dim=-1)
    # The correct class is the support image at the same index.
    logits = 10.0 * (query @ support.T)
    targets = torch.arange(ways, device=device)
    loss = F.cross_entropy(logits, targets)
    correct = (logits.argmax(dim=1) == targets).sum().item()
    return loss, correct, ways


def run_episodic_epoch(model, dataset, classes, ways, episodes, seed, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    rng = random.Random(seed)
    total_loss = total_correct = total_queries = 0
    with torch.enable_grad() if training else torch.no_grad():
        for _ in range(episodes):
            images = make_episode(dataset, classes, ways, rng)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, correct, count = episodic_loss(model, images, ways, device)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * count
            total_correct += correct
            total_queries += count
    return total_loss / total_queries, total_correct / total_queries



def create_ordinary_loaders(module, dataset, batch_size, train_fraction=1.0):
    functions = {
        "numerical": "load_NumericalSets",
        "modelnet": "load_ModelNet40",
        "cifar": "load_CIFAR100",
        "gaussian": "load_GaussianMixtures",
    }

    train_set, test_set = getattr(module, functions[dataset])()

    # Gaussian Mixtures originally returns one cluster label per point.
    # Convert these into one label per set: 0, 1, 2, or 3.
    if dataset == "gaussian":
        def convert_labels(source):
            points, labels = source.tensors

            # Count the distinct clusters actually represented in each set.
            cluster_counts = (
                torch.nn.functional.one_hot(labels.long(), num_classes=4)
                .any(dim=1)
                .sum(dim=1)
                .long()
            )

            targets = cluster_counts - 1
            return torch.utils.data.TensorDataset(points, targets)

        train_set = convert_labels(train_set)
        test_set = convert_labels(test_set)

    n_val = max(1, int(len(train_set) * 0.1))

    train_set, val_set = random_split(
        train_set,
        [len(train_set) - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    # Use a fixed, nested subset of the training split.
    # Validation and test sets remain unchanged.
    n_train = max(1, int(len(train_set) * train_fraction))

    indices = torch.randperm(
        len(train_set),
        generator=torch.Generator().manual_seed(SEED)
    )[:n_train].tolist()

    train_set = torch.utils.data.Subset(train_set, indices)

    print(f"Training fraction: {train_fraction:.0%}")
    print(f"Training examples: {len(train_set)}")
    print(f"Validation examples: {len(val_set)}")
    print(f"Test examples: {len(test_set)}")

    train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train, val, test



def create_omniglot_data(module, ways):
    train_dataset, test_dataset = module.load_Omniglot()
    all_classes = omniglot_class_indices(train_dataset)
    ids = sorted(all_classes)
    random.Random(SEED).shuffle(ids)
    split = int(0.9 * len(ids))
    train_classes = {k: all_classes[k] for k in ids[:split]}
    val_classes = {k: all_classes[k] for k in ids[split:]}
    test_classes = omniglot_class_indices(test_dataset)
    for label, classes in (("train", train_classes), ("val", val_classes), ("test", test_classes)):
        if len(classes) < ways:
            raise ValueError(f"Only {len(classes)} {label} classes; need {ways}")
    return train_dataset, test_dataset, train_classes, val_classes, test_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(CONFIG), default="numerical")
    parser.add_argument("--model", choices=["all", *MODEL_CLASSES], default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of the training split to use (0 < fraction <= 1)"
    )   
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-module", default="load_data", help="Dataset loader filename without .py")
    parser.add_argument("--episodes", type=int, default=100, help="Omniglot training episodes per epoch")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Omniglot validation/test episodes")
    parser.add_argument("--ways", type=int, default=5, help="Omniglot number of classes per episode")
    args = parser.parse_args()
    if not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction must be between 0 and 1")
    if min(args.epochs, args.batch_size, args.episodes, args.eval_episodes) < 1 or args.ways < 2:
        parser.error("Epochs, batch size, episodes must be positive; ways >= 2")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    module = importlib.import_module(args.data_module)
    if args.dataset == "omniglot":
        train_data, test_data, train_classes, val_classes, test_classes = create_omniglot_data(module, args.ways)
    else:
        train_loader, val_loader, test_loader = create_ordinary_loaders(
            module,
            args.dataset,
            args.batch_size,
            args.train_fraction
        )

    names = list(MODEL_CLASSES) if args.model == "all" else [args.model]
    summary = []
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    for name in names:
        set_seed(SEED)
        model = make_model(name, args.dataset).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*65}\n{name.upper()} / {args.dataset.upper()} / {n_params:,} parameters\n{'='*65}")
        best_loss, best_state = float("inf"), None
        start = time.time()
        history = []
        for epoch in range(1, args.epochs + 1):
            if args.dataset == "omniglot":
                train_loss, train_metric = run_episodic_epoch(
                    model, train_data, train_classes, args.ways, args.episodes,
                    SEED + epoch, device, optimizer)
                val_loss, val_metric = run_episodic_epoch(
                    model, train_data, val_classes, args.ways, args.eval_episodes,
                    SEED + 10000, device)
            else:
                train_loss, train_metric = run_ordinary_epoch(model, train_loader, args.dataset, device, optimizer)
                val_loss, val_metric = run_ordinary_epoch(model, val_loader, args.dataset, device)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
            scheduler.step()
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "train_metric": train_metric, "val_loss": val_loss,
                            "val_metric": val_metric})
            metric_label = "MAE" if args.dataset == "numerical" else "accuracy"
            print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} | "
                  f"val loss {val_loss:.4f} | val {metric_label} {val_metric:.4f}")

        suffix = f"_fraction_{int(args.train_fraction * 100)}"
        history_path = Path("results") / f"{args.dataset}_{name}{suffix}_history.csv"
        with history_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_metric",
                                                   "val_loss", "val_metric"])
            writer.writeheader()
            writer.writerows(history)
        print(f"Saved learning curve data: {history_path}")

        model.load_state_dict(best_state)
        if args.dataset == "omniglot":
            test_loss, test_metric = run_episodic_epoch(
                model, test_data, test_classes, args.ways, args.eval_episodes,
                SEED + 20000, device)
        else:
            test_loss, test_metric = run_ordinary_epoch(model, test_loader, args.dataset, device)
        seconds = time.time() - start
        path = Path("checkpoints") / f"{args.dataset}_{name}{suffix}.pt"
        torch.save({"state_dict": model.state_dict(), "dataset": args.dataset,
                    "model": name, "test_loss": test_loss, "test_metric": test_metric,
                    "parameters": n_params, "epochs": args.epochs}, path)
        display = f"MAE {test_metric:.4f}" if args.dataset == "numerical" else f"accuracy {test_metric*100:.2f}%"
        print(f"TEST: loss {test_loss:.4f}, {display} | {seconds:.1f}s | saved {path}")
        summary.append((name, test_loss, test_metric, seconds))

    print("\nFINAL COMPARISON")
    for name, loss, metric, seconds in summary:
        display = f"MAE {metric:.4f}" if args.dataset == "numerical" else f"accuracy {metric*100:.2f}%"
        print(f"{name:10s} | loss {loss:.4f} | {display} | {seconds:.1f}s")


if __name__ == "__main__":
    main()

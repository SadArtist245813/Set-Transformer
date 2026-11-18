# Load all dataset for training/testing

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms


def make_dataloaders(load_dataset, *, batch_size=32, num_workers=0, **dataset_options):
    """Call a load_* function and batch its training and test splits."""
    train_set, test_set = load_dataset(**dataset_options)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


#Omniglot
def load_Omniglot():
    # Define transformations: resize to 96x96 and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load background (train) and evaluation (test) sets
    train_dataset = datasets.Omniglot(root="./data", background=True, download=True, transform=transform)
    test_dataset = datasets.Omniglot(root="./data", background=False, download=True, transform=transform)

    print(f"Training samples: {len(train_dataset)}, Classes: {len(train_dataset._characters)}")

    return train_dataset, test_dataset

# RGB image datasets
def _rgb_transform():
    return transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

def load_CIFAR10(root="./data", *, download=True):
    """Return official training and test splits of (RGB image, class ID)."""
    transform = _rgb_transform()
    train = datasets.CIFAR10(root, train=True, download=download, transform=transform)
    test = datasets.CIFAR10(root, train=False, download=download, transform=transform)
    return train, test


def load_CIFAR100(root="./data", *, download=True):
    """Return official training and test splits of (RGB image, class ID)."""
    transform = _rgb_transform()
    train = datasets.CIFAR100(root, train=True, download=download, transform=transform)
    test = datasets.CIFAR100(root, train=False, download=download, transform=transform)
    return train, test


def load_CelebA(root="./data", *, download=True):
    """Return official training and test splits of (RGB image, identity ID)."""
    options = dict(target_type="identity", download=download, transform=_rgb_transform())
    return (
        datasets.CelebA(root, split="train", **options),
        datasets.CelebA(root, split="test", **options),
    )

# ModelNet40
try:
    from torch_geometric.datasets import ModelNet
    from torch_geometric.transforms import NormalizeScale, SamplePoints
except ImportError:
    ModelNet = None
    NormalizeScale = None
    SamplePoints = None

class _ModelNetPointClouds(Dataset):
    def __init__(self, meshes):
        self.meshes = meshes

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, index):
        sample = self.meshes[index]
        return sample.pos, sample.y.item()

def load_ModelNet40(root="./data", *, num_points=1024):
    """Return official splits of (sampled [num_points, 3] point cloud, class ID).

    Requires torch-geometric. Meshes are downloaded and cached on first use.
    """

    if num_points < 1:
        raise ValueError("num_points must be positive")
    path = str(Path(root) / "modelnet40")
    transform = SamplePoints(num_points)
    normalize = NormalizeScale()
    train = ModelNet(path, name="40", train=True, transform=transform, pre_transform=normalize)
    test = ModelNet(path, name="40", train=False, transform=transform, pre_transform=normalize)
    return _ModelNetPointClouds(train), _ModelNetPointClouds(test)

# GaussianMixtures
def load_GaussianMixtures(
    *, train_size=10000, test_size=2000, set_size=100, max_clusters=4, seed=0
):
    """Return splits of ([set_size, 2] points, [set_size] cluster IDs).

    Each set samples its own centers and mixture weights. Cluster IDs are local
    to each set.
    """
    if not 1 <= max_clusters <= set_size:
        raise ValueError("max_clusters must be between 1 and set_size")
    generator = torch.Generator().manual_seed(seed)

    def make_split(size):
        # Draw a number of active components and a mixture for each set.
        counts = torch.randint(1, max_clusters + 1, (size, 1), generator=generator)
        active = torch.arange(max_clusters) < counts
        weights = (1 - torch.rand(size, max_clusters, generator=generator)) * active
        labels = torch.multinomial(weights, set_size, replacement=True, generator=generator)
        centers = 4 * torch.randn(size, max_clusters, 2, generator=generator)
        points = centers[torch.arange(size)[:, None], labels]
        points += torch.randn(size, set_size, 2, generator=generator)
        return TensorDataset(points, labels)

    return make_split(train_size), make_split(test_size)

# NumericalSets
def load_NumericalSets(*, train_size=10000, test_size=2000, set_size=100, seed=0):
    """Return splits of ([set_size, 1] standard normal values, scalar maximum)."""
    if set_size < 1:
        raise ValueError("set_size must be positive")
    generator = torch.Generator().manual_seed(seed)

    def make_split(size):
        values = torch.randn(size, set_size, generator=generator)
        return TensorDataset(values.unsqueeze(-1), values.amax(dim=1))

    return make_split(train_size), make_split(test_size)

# Set Transformer

A PyTorch implementation of the [Set Transformer](https://arxiv.org/abs/1810.00825)
from Lee et al. (ICML 2019), with experiments comparing attention-based models
and an MLP baseline on learning from sets.

The project includes a full Set Transformer, ISAB + PMA and SAB + PMA variants,
a standard Transformer, and an MLP baseline. Experiments cover numerical
maximum-value regression, Gaussian mixture cluster counting, and image
classification using sets of image patches. The inference API targets CIFAR-10,
with one score for each of its 10 image classes. Saved notebooks and figures are
available in `notebooks/` and `results/`.

## Installation

Install Git and Python 3.10 or newer. Run the following commands in a terminal:

```bash
git clone https://github.com/SadArtist245813/Set-Transformer.git
cd Set-Transformer
```

Install the dependencies:

```bash
python -m pip install torch torchvision numpy pandas matplotlib jupyter
```

## Run the models

```bash
python test_models.py
```

The script imports `load_model` and `sample_batch` from `model_api.py`, loads
the symmetric Set Transformer and the MLP baseline, runs each model on the
sample inputs, and prints the output shapes.

The CIFAR-10 inference API requires these files beside the source modules:

```text
model_assets/
    symmetric.pt
    baseline.pt
    sample_batch.pt
```

`symmetric.pt` and `baseline.pt` must contain the corresponding trained
CIFAR-10 model weights with 10 output classes. They may contain a plain state dictionary or a
checkpoint dictionary with a `state_dict` entry. `sample_batch.pt` must contain
real preprocessed test inputs shaped `(B, 144, 194)`, either as a tensor or a
dictionary with an `inputs` tensor.

The input representation resizes each image to 96 x 96 pixels and divides it
into 144 patches. Each patch contains 192 RGB values and two spatial coordinates,
giving 194 features. Both classifiers return `(B, 10)` class scores.

**These artifacts are not currently included.** Until they are supplied,
`test_models.py` raises a missing-file error. It does not substitute untrained
models or synthetic examples. After the artifacts are added, both models use
the GPU when available, otherwise the CPU. With eight test examples, the
expected output is:

```text
symmetric torch.Size([8, 10])
baseline torch.Size([8, 10])
```

Printing shapes checks inference. It does not train the models or calculate
test accuracy.

## Project files

| File or folder | Purpose |
| --- | --- |
| `set_transformer.py` | Full Set Transformer and attention variants |
| `transformer.py` | Standard Transformer used in the experiments |
| `baseline.py` | MLP baseline |
| `load_data.py` | Dataset loaders and synthetic dataset generators |
| `train.py` | Training and evaluation entry point |
| `model_api.py` | Trained-model and sample-loading interface |
| `test_models.py` | Inference script for the two API model kinds |
| `notebooks/` | Experiment notebooks and recorded results |
| `results/` | Saved experiment figures |
| `imformation/` | Reference paper and supporting materials |

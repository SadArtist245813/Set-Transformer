# Set Transformer

A PyTorch implementation of the [Set Transformer](https://arxiv.org/abs/1810.00825)
from Lee et al. (ICML 2019), with experiments comparing attention-based models
and an MLP baseline on learning from sets.

The project includes a full Set Transformer, ISAB + PMA and SAB + PMA variants,
a standard Transformer, and an MLP baseline. Experiments cover numerical
maximum-value regression, Gaussian mixture cluster counting, and image
classification using sets of image patches. Saved notebooks and figures are
available in `notebooks/` and `results/`.

## Authors

Jimmy Finnegan - Programming and HPC Configuration
Vidhula Oviya Vasanthan - Presentation Management
Justin Hortopanu - PyTorch Modeling, Project Management and Analysis  

## Installation

Install Git and Python 3.10 or newer. Run the following commands in a terminal:

```bash
git clone https://github.com/SadArtist245813/Set-Transformer.git
cd Set-Transformer
```

Install the dependencies:

```bash
python -m pip install .
```

## Run the models

```bash
python test_models.py
```

The script imports `load_model` and `sample_batch` from `model_api.py`, loads
the symmetric Set Transformer and the MLP baseline, runs each model on the
sample inputs, and prints the output shapes.

Printing shapes checks inference. It does not train the models or calculate
test accuracy.

## Using the inference API

The project provides two pretrained CIFAR-10 classifiers through `model_api.py`:

- `symmetric`: Full Set Transformer
- `baseline`: MLP baseline

Both models can be loaded and evaluated without downloading datasets or retraining.

```python
from model_api import load_model, sample_batch

inputs = sample_batch()

symmetric = load_model("symmetric")
baseline = load_model("baseline")

print(symmetric(inputs).shape)
print(baseline(inputs).shape)
```

Expected output:

```text
torch.Size([8, 10])
torch.Size([8, 10])
```

The API automatically uses a GPU when CUDA is available and otherwise runs on the CPU.

The three required files (`symmetric.pt`, `baseline.pt`, and `sample_batch.pt`) are included in the installation package. The package was tested successfully from outside the project directory.

## Run on the a HPC

Every test was ran using 1 cpu, 1 gpu 32gb of ram.
The notebooks folder has the files used to run the different tests.

## Training and reproducibility

The main training entry point is `train.py`.

For example, to train all five architectures on CIFAR-10 for 20 epochs:

```bash
python train.py --dataset cifar --model all --epochs 20
```

The project experiments were conducted using high-performance computing resources. Training requires access to the relevant datasets and a compatible Python environment.

Experiment notebooks and saved figures are available in `notebooks/` and `results/`.

The pretrained inference API does not require running the training scripts.

## Results

Five architectures were evaluated across Numerical Sets, Gaussian Mixtures,
CIFAR-10, CIFAR-100, and ModelNet40.

ModelNet40 test accuracy after 20 epochs:

| Model | Test Accuracy |
|---|---|
| Standard Transformer | 82.70% |
| Set Transformer | 80.83% |
| ISAB + PMA | 80.96% |
| SAB + PMA | 79.78% |
| MLP Baseline | 34.89% |

Additional results, including training loss, validation loss,
and test accuracy comparisons, are available in `results/`.

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
| `torch_canvas/` | Reference to Torch Canvas model and the code |
| `pyproject.toml` | Package installation and dependencies |
| `model_assets/` | Pretrained model weights and sample inputs |
| `checkpoints/` | Saved training checkpoints |

## Reference

Lee et al. (2019). [Set Transformer: A Framework for
Attention-based Permutation-Invariant Neural Networks](https://arxiv.org/abs/1810.00825).
International Conference on Machine Learning (ICML).

https://www.jessetnroberts.com/torch-canvas/#zip=3VvdbuO4FX4VgbfVeEWKEiVfNYtedNAOMMFsuwsUC0OOaEdrW_RIsifZQYBe9n7v-iK976PskxTUT3goHzlO4p9pcuVIPCS_j4eH5xPJryRPVpKMySdZOT8USV7OVLGShfPOuSrLbJ6vZF4Rl-QqlSUZ_-MryVIyJjkNOHFJdb_Wtlm-3uhC66RIViUZf-0q_Ym4pLxN6kLSdajnOlSbZdVSPyIPLrkjY88l92QsPO_BNdUHpvrpUt0srOqbJ2Py_tPV95P3betJMdcvH1ySv8_JmLqkkGuZVPXPtkNZmUxhD3QFDnX--x_HZ06Wp5ubLJ87a5XlVUlcsk0K3U2XLLNp15HyO13JpMY8-qVUuYUitFGIg1H8OUtTmT8DBuvDYC-BcVs3u4sjsHHEh-D4-OFqYjnNIVDWqwQA-fjhSqNYy2K1qZIqU7mT5dukyJK6QhzIWqllls8n0_tJUlUy12a7iLiFKKSHINID82xEPRerPWyw76Ws9nXatzvtn7LTrNdp9tJO0ziCnQYzeZnlMimsXmc5GRPKIuIStan0b-KSaZaUZFwVG-mSLM_041TOks1Sw9EPJkkxbzr2qyzUBJbXvyetUY1INj9aoIWcF7IsM5UDuH-t--VY71roFjB7NMDkVptqIASuC5lmN5VMJ6vkbij2gWpvNl_uFwtAmSr1FL4jY-a3hWErM03fqpTEJSuVaoN5tq3jyDKZyuWkXClV3WZ5y1Y2z1UhJ1meyrvmSSqXVaJpH3nEJV9kNr_tRmGVlIvJulB65L0RDbpHdUctKKD7X_gsEbtLg-5_3LLIPISn-95S0V8pHn52iUzncBGSNAiJS2aFWoElSZn1Y62KCpIraRBZBgEwEKhB6FkGAhjEuAGzDGJjUIccxIBDgyYuKTPdEQMLdBMTlJlqiIEFOgSgQxT0zXZRzjYDNo_-uWt2d7eoCerMjC9gllSP6LxQm7Ue0p9dUt6vpmrZRqom0mj_-HD1_eQvf6d1rNDtdP8_zujmgaZ9XiTrW-BT5s1O-nJA8vJ5I4ueV1afXaeJVb1pTKNgxDzwx3azAXZAmzcqr-Rd1Wt1gaZNvvBHIgZ_wW6b_jNjb_37hNF3djP5DJC0_1qhloZ01CAJWGiQgFXkViZpaQFpnowJh0vYeplVTlfWboLxtgkeMNOEOICss1K1sKla9HEwEWNUxUekyjRhUUXpt8bV1uZq2wfixwLjivpHJIv7qF9R4LuVllhrVS_Ypp2u7h-6t85C3u8ZC1_ADAuKjCxXa6tqtSZj8kfi6tBStvQVckvGs2RZSgPrWke73__520LeO-WNKuS-5j0Owpr3ZPPfgeYpHfnUF17E4yiOuKgXPKxDf8q2WSqd6b3z-7_-3XjQQHeYD6MsCLPJjR1k62SpVLOqScXSTC9T76xkvXnpqK0snhgDGgA3YvwYY_BjnX0522S52Ud_BHyLPT34f3i64as0ddaF-kXqRNXpFr6h5sHyAsZ-JYs57tUf9Junwsu7ENDps29v3VJ2gFF9KCKKUCj8CX8s5HKDO-NwC3AIwmM5QKNhnEKWWbpJlntaB_7ng8VmWASp-nEvfzEV6qmEpPdWYvyYRqJJMUyJTUKMpsMwGWaPRfHsHyb_JpHFE3-G1kvxpN-CBlL-gYwflAaCAs_2IT6Q6uMAY0usYKWpKc0oLG34qFeAnbqZDwnxTGmGlg5gacNfHVh3S1sSyMdKw37jlDCUEgYpYRwrDar2ISXMUOKjlPiQEt9Q4qOU-JAS31Dio5T4YqDuEC0NKfE5Vhqg5HAeNAFHPU7-ru5hNffQfoxopZ31KWmVTCeLLXv8iKQl27Ut9q57Wu96SOpdY0qPv0bqIUJPCFt0Rbuii79O6WH60g_DEUdaOkTf_T-oO34GeccP0XdnTipeqPD4GSRecIjEOzNdLxR5wRlEXnBakRdcVuSF35bICy8i8sJLibzwoiJPnEjkibcj8sSJRZ64rMgTpxR5UApxI4U4KoU4TPy5Sfw5qoU4zHK5yaA5mvkHMMvlJpfnqN4LGF53gAq-AKIMDMoARWlta3Gj-PBNLWtPixt9g-9oBVDfBGhpkPmHUN-A3a8Q1Tch1Deh0SAhqm9CqG9Cw2CI6htr1yvwsdKw3wOc4BteIeQk5FhpULeAnISGE4FyIiAnwnAiUE4E5ESAzxMoJ0IM1I1qPgE5ERwrDVBG1rcPo_nEcTTfZ0vy6dAOJF8T6Y2ya_5HJF9bsCf5oqPv7gkxQnRedAKdJ3y0pbezjxedQelFb0fpRWdQevHbUXrxGZRefFqlF19W6VHv25J61LuI1qPepcQe9S6q9ig9kdyj9O3oPUpPLPgovazio_SEki-CYigyYihCxVAEE__IJP4RvrsHk9zIJNARmvjHMMmNTCofoZIvZnjdMSr5YogyNihjFGVsoTSSL0ZRxhZKI29iHCWUNzFaGiT-1IP6JgYnHj1U4FAPKpx2-VAmdCMGgWUANks9VOZQz9oA9dHyFoIBeprQitRvbYN6HDWADVBrJ9QDHFGcI2pxRAFHFOeIWhxRuKGMc0TFYAuoGqTWRnEb0HYMIGhmHYWlITA4iiakLHrUhOBiR6cLrbsebfyxnvX1oW3QPwDKDhGJvfsrOb4lSANsb47Ck0h1zVYLXbXUdXzWqcF2rcpVsaojNVjfvFE9UTpsexYUFnJMQ1J20O0Bc7y2d22AYdcGVsnUA115b189ccrNapUU2a_SqW6lU8rhOxz9fWDNa9ztedon9lh4KI7rZ8CwkkZZOXIp9fWJ0qkvOqROpRoMDSKYzO3_utEAQQ4FspevsIYZrRywg_HMOmACzlM0XoCcXrHiDaOogRUNxKABHm-sUxitM-4YwBZ8K94wEG_Y6-MNcocLXsOygs7j1SwYWB4fomHHmPTjjv_CuIN-l2JsFMFT5xSZ9v6lwpCIRyHWn0PDUNOXc0chuAppFF73Qa53bPjQIPQsFMcMQoNA7DD0igN8psqhMORbYQicYWq8YDdIWGee2rmyY2AFCTFogIch6-RT64w7BrAF6_ATBaef6BGOP2F3MHvXKLtQtHO7sh2aj-b241V3Jw8JSwPFduLTQeeljhmf-KHxiT47PH3aG57CEdqdk4Qn0JErZ5bdydRZyqTIu08zzipZyNJpb7I6qkhlccDNV2yaeyMfiVc8fMU076rUkQOb5tYeLgWboRTfxaXcmuZg1xcaWJPQmuZg47cB9rpJuPf-cO82bTcbdy7Zmmv0e6dh__3u_IueP__o0PzzBebhB13mfoWH_62Uzk_OTBW1b2eydOsvr06SpztfQp906A6D7c4Be7E7dxUOObN1DKAdDmWIQ-6U0gMMoDMH1scADk90s1c7M3Y3-8El8q6p9uvDw_8A

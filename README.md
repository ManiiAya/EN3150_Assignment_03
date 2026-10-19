# EN3150 Assignment 03 — Resource-Constrained CNN for Edge Image Classification

**Module:** EN3150 — Pattern Recognition  
**Department:** Electronic and Telecommunication Engineering, University of Moratuwa  
**Group:** LogicLoop

## Team Members

| Index No. | Name |
|---|---|
| 230065H | N.B.G.M. Ayanaja |
| 230248X | W.V.M.J. Himasara |
| 230266B | J.C.R.N. Jathunwaththa |
| 230280L | J.A.P.R. Jayasinghe |

## Project Overview

This project investigates the trade-off between **image-classification accuracy, model memory, and computational cost** for resource-constrained edge devices. Four convolutional neural networks (CNNs) are evaluated on the **EuroSAT RGB** land-use/land-cover dataset:

- **Model A — Standard CNN:** A custom baseline built with standard convolutions and max-pooling.
- **Model B — Lightweight CNN:** A custom depthwise-separable CNN with **59,301 trainable parameters**, below the assignment's 100,000-parameter limit.
- **MobileNetV2:** An ImageNet-pretrained lightweight network fine-tuned for EuroSAT.
- **SqueezeNet 1.1:** An ImageNet-pretrained compact network fine-tuned for EuroSAT.

The custom models are trained from scratch. Both pretrained networks are fine-tuned **end to end**, with their classifiers replaced to predict the ten EuroSAT classes.

## Repository Structure

```text
EN3150_Assignment_03/
├── models/
│   ├── __init__.py
│   ├── lightweight_cnn.py    # Model B
│   ├── standard_cnn.py       # Model A
│   └── sota_models.py        # Additional SOTA model definitions
├── notebooks/
│   ├── section_01.ipynb     # Custom CNNs, optimizer study, evaluation
│   └── section_02.ipynb     # Pretrained networks and final comparison
├── data_prep.py             # Seeded data split, normalization, DataLoaders
├── utils.py                 # Training, evaluation, parameter count, plots
├── requirements.txt
├── .gitignore
└── README.md
```

Training the Section 2 models also creates `checkpoints/` (best model weights) and `results/section02/` (CSV metrics, histories, and metadata). These are **generated outputs**, not required source files. The main submission notebooks are `section_01.ipynb` and `section_02.ipynb`.

## Dataset and Preprocessing

**Dataset:** [EuroSAT](https://github.com/phelber/EuroSAT), RGB version (Sentinel-2 satellite imagery).  
**Samples:** 27,000 RGB images across 10 categories.  
**Input:** `3 × 64 × 64` pixels.  
**Classes:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, and SeaLake.

The shared `create_dataloaders()` function in `data_prep.py` performs the following:

| Subset | Ratio | Images |
|---|---:|---:|
| Training | 70% | 18,900 |
| Validation | 15% | 4,050 |
| Test | 15% | 4,050 |

- Uses **random seed 26** for the dataset split.
- Resizes images to `64 × 64`, converts them to PyTorch tensors, and normalizes each RGB channel using **training-set statistics only**.
- Uses the same deterministic splitting procedure for both notebooks; no data augmentation is applied.
- Keeps the test subset separate for final evaluation.

Training-set RGB statistics from the reported experiments:

| Channel | Mean | Standard deviation |
|---|---:|---:|
| Red | 0.3438 | 0.2026 |
| Green | 0.3800 | 0.1370 |
| Blue | 0.4077 | 0.1156 |

### Dataset setup

Download/extract the **EuroSAT RGB** images from the dataset source and place the class directories at the following path:

```text
data/
└── EuroSAT_RGB/
    ├── AnnualCrop/
    ├── Forest/
    ├── HerbaceousVegetation/
    ├── Highway/
    ├── Industrial/
    ├── Pasture/
    ├── PermanentCrop/
    ├── Residential/
    ├── River/
    └── SeaLake/
```

Each class folder should contain its corresponding images. The code expects exactly **one class-directory level** below `data/EuroSAT_RGB/`. The dataset is not included in the repository.

## Installation and Reproduction

**Requirements:** Python, PyTorch/TorchVision, NumPy, pandas, scikit-learn, Matplotlib, TorchInfo, and a Jupyter-compatible notebook environment. See `requirements.txt` for the project's dependency list. A CUDA-enabled GPU is recommended for fine-tuning, but the code can select CPU when CUDA is unavailable.

From the cloned repository root:

```bash
python -m venv .venv
# Activate .venv for your operating system, then:
python -m pip install -r requirements.txt
```

Open the notebooks **with the working directory set to `notebooks/`** (important for Section 1's relative paths):

```bash
cd notebooks
jupyter notebook
```

If the `jupyter` command is not installed, install it with `python -m pip install notebook` first.

Run the notebooks in the following order:

1. **`section_01.ipynb`:** Load the dataset; define/evaluate Models A and B; compare Adam, SGD, and SGD with momentum; generate loss/accuracy curves and confusion matrices.
2. **`section_02.ipynb`:** Initialize ImageNet-pretrained MobileNetV2 and SqueezeNet 1.1; train both networks; select the checkpoint with the highest validation accuracy; evaluate on the held-out test data; produce the model-comparison outputs.

You can also open the notebooks in VS Code or run them with a Colab GPU after making the repository files and dataset available. The **same `data_prep.py` and `utils.py`** are used; no separate preprocessing pipeline is needed for Colab. Pretrained weights may be downloaded by TorchVision on first use.

> **Reproducibility:** The dataset directory structure and ordering, package versions, device, and random states can affect results. Re-running training may produce small numerical differences. The figures below are results from the experiments reported in the submitted PDF, not guaranteed values from every new run.

## Training Configuration

| Configuration | Custom CNNs (A and B) | Pretrained networks |
|---|---|---|
| Initialization | From scratch | TorchVision ImageNet-1K V1 |
| Optimizer | Adam | Adam |
| Learning rate | `1e-3` | `1e-4` |
| Batch size | 64 | 64 |
| Epochs | 20 | 20 |
| Loss | Cross-entropy | Cross-entropy |
| Trainable layers | Entire network | Entire network |
| Model selection | As implemented in Section 1 | Highest validation accuracy checkpoint |

Section 1 additionally compares standard SGD and SGD with momentum against Adam.

## Results

The following values correspond to the **reported held-out EuroSAT test experiments**. Model A and Model B figures come from Section 1; MobileNetV2 and SqueezeNet 1.1 figures come from the latest Section 2 run.

| Model | Test accuracy | Parameters | FP32 weight storage | Analytical MACs/image |
|---|---:|---:|---:|---:|
| Model B — Lightweight CNN | 84.86% | 59,301 | 0.226 MiB | 2.06 M |
| Model A — Standard CNN | 92.30% | 914,122 | 3.487 MiB | 60.70 M |
| SqueezeNet 1.1 | 95.36% | 727,626 | 2.776 MiB | 17.023 M |
| **MobileNetV2** | **97.11%** | 2,236,682 | 8.532 MiB | 24.461 M |

### SOTA fine-tuning results

| Metric | MobileNetV2 | SqueezeNet 1.1 |
|---|---:|---:|
| Test accuracy | 97.11% | 95.36% |
| Macro precision | 97.00% | 95.47% |
| Macro recall | 96.91% | 94.93% |
| Macro F1-score | 96.95% | 95.13% |
| Best validation accuracy | 98.12% | 95.68% |
| Best checkpoint epoch | 20 | 19 |
| Average training time per epoch | 9.20 s | 7.31 s |

**Interpretation:** Model B minimizes parameter storage and analytical operations, while MobileNetV2 achieves the highest test accuracy. SqueezeNet 1.1 offers an intermediate accuracy–complexity trade-off. Parameter storage is **not** the complete inference RAM requirement; MACs are analytical estimates, **not** measured latency or energy consumption. No on-device deployment, quantized-model evaluation, or distribution-shift robustness test was performed.

### Generated Section 2 outputs

After running `section_02.ipynb`, the following files are written under `results/section02/`:

```text
results/section02/
├── sota_metrics.csv
├── model_comparison.csv
├── MobileNetV2_history.csv
├── SqueezeNet1_1_history.csv
└── experiment_metadata.json
```

The notebook also displays training/validation plots and row-normalized confusion matrices. The **complete analysis, figures, and trade-off discussion** are provided in the separately submitted assignment report.

## Submission and Notes

- Submit the **PDF report and code separately** as required by the assignment, using the specified group-number filename format.
- The notebooks rely on `data_prep.py`, `utils.py`, and the relevant `models/` modules; keep those files with the submission.
- Keep EuroSAT images and saved checkpoints out of version control due to size (verify that `.gitignore` includes `data/EuroSAT_RGB/` and `checkpoints/`). Do **not** commit GitHub tokens, passwords, or other credentials.
- The Git history records the team's implementation work. If the repository remains private, the course staff will need appropriate GitHub access to inspect it.

For the course requirements and detailed experimental conclusions, refer to the final **EN3150 Assignment 03 report**.

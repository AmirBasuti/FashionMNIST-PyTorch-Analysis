# Fashion-MNIST: Neural Networks in PyTorch

A coursework notebook exploring image classification and reconstruction with PyTorch. It implements an MLP, a convolutional classifier and a convolutional autoencoder, then examines training curves, classification errors, learned filters and reconstructed images.

[Open the analysis notebook](HW2-AmirMohammadBasuti-610304080.ipynb)

## Inside the notebook

- MLP and CNN classification with batch normalization and dropout.
- Adam and AdamW training experiments, including a weight-decay configuration.
- A convolutional autoencoder with a 64-dimensional bottleneck and reconstruction loss.
- Learning curves, confusion matrices, per-class reporting, misclassified examples and feature-map visualizations.

The notebook loads Fashion-MNIST through torchvision, divides the provided training set into 80% training and 20% validation subsets, and uses the provided test set for evaluation. Images are normalized to the range expected by the models. The code selects CUDA when available and otherwise selects CPU.

## Reading the results

The saved notebook contains historical outputs from the coursework. They have not been rerun or independently reproduced for this documentation, and they should not be read as a new benchmark or a controlled ablation study.

Two implementation details matter when interpreting them:

- Training and validation currently share the same dataset object, so the random training augmentation also affects validation.
- The training helper saves the best state with a shallow `state_dict().copy()`. This does not independently freeze its tensors and may fail to preserve the intended best epoch.

The comparison also changes optimizer choice alongside weight decay and explores multiple configurations against the test set. These limits need to be resolved before making a stronger claim about checkpoint selection, repeatability or the isolated effect of regularization.

## Opening and running locally

The repository currently contains the notebook and this README. Dependencies are not locked, and no automated test suite or CI workflow is included. The following setup is derived from the imports and has not been executed as a clean-environment reproduction:

```sh
git clone https://github.com/AmirBasuti/FashionMNIST-PyTorch-Analysis.git
cd FashionMNIST-PyTorch-Analysis
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux. Then install a compatible PyTorch/torchvision pair and the remaining notebook dependencies:

```sh
python -m pip install torch torchvision torchinfo numpy matplotlib seaborn scikit-learn tqdm notebook
jupyter notebook
```

Open the notebook and inspect the cells before running them in order. Its dataset-loading cells use `download=True`, write under `data/`, and require network access if the dataset is not cached. Training the models is substantive computation; opening saved outputs does not require retraining. No runtime, memory or cross-platform compatibility guarantee is established here.

## Context and terms

This is a Machine Learning coursework project. The existing notebook and saved results retain their original assignment context. Fashion-MNIST is an external dataset, available from its [original project](https://github.com/zalandoresearch/fashion-mnist).

No standalone software license is currently included in this repository. This README does not add a license or change the terms of the dataset or dependencies.

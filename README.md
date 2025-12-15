# HW2 - Neural Networks on Fashion-MNIST

**Author:** AmirMohammad Basuti  
**Student ID:** 610304080  
**Course:** Machine Learning (Dr. Sajedi)

---

## 📋 Project Overview

This project implements and compares different neural network architectures for image classification on the Fashion-MNIST dataset. The goal is to explore the effectiveness of various models, optimizers, and regularization techniques in achieving high classification accuracy.

### Objectives
- Implement an **MLP (Multilayer Perceptron)** as a baseline classifier
- Implement a **CNN (Convolutional Neural Network)** that preserves spatial features
- Implement an **AutoEncoder** for unsupervised representation learning
- Compare different optimizers and regularization techniques
- Analyze the impact of hyperparameters on model performance

---

## 📊 Dataset: Fashion-MNIST

| Property | Value |
|----------|-------|
| Training Samples | 60,000 (48,000 train / 12,000 validation) |
| Test Samples | 10,000 |
| Image Size | 28 × 28 pixels (grayscale) |
| Number of Classes | 10 |
| Channels | 1 |

### Classes
| Label | Class Name |
|-------|------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## 🏗️ Model Architectures

### 1. MLP (Multilayer Perceptron)

A fully-connected neural network that processes flattened image vectors.

```
Input (784) → FC(256) → BN → ReLU → Dropout(0.2)
           → FC(128) → BN → ReLU → Dropout(0.2)
           → FC(10) → Output
```

**Key Features:**
- Input flattening: 28×28 → 784
- Configurable hidden layer sizes
- Batch Normalization for training stability
- Dropout for regularization
- ~236K trainable parameters

### 2. CNN (Convolutional Neural Network)

A convolutional architecture that preserves spatial structure.

```
Input (1×28×28) 
  → Conv Block 1: Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool → Dropout
  → Conv Block 2: Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout
  → Flatten → FC(128) → BN → ReLU → Dropout → FC(10) → Output
```

**Key Features:**
- Automatic flatten dimension calculation via dummy forward pass
- Two convolutional blocks with increasing filter depth
- 3×3 kernels with padding to preserve dimensions
- 2×2 MaxPooling for spatial reduction
- ~340K trainable parameters

### 3. AutoEncoder CNN

An encoder-decoder architecture for unsupervised representation learning.

```
Encoder: Input (1×28×28) → Conv(32, stride=2) → Conv(64, stride=2) → FC(64)
Decoder: FC(3136) → ConvTranspose(32) → ConvTranspose(1) → Output (1×28×28)
```

**Key Features:**
- 64-dimensional latent space (bottleneck)
- Strided convolutions for downsampling
- Transposed convolutions for upsampling
- Tanh activation for output (matches normalized input range)
- ~157K trainable parameters

---

## 🔧 Techniques & Technologies

### Data Augmentation
Applied only during training to improve generalization:

| Transform | Description |
|-----------|-------------|
| `RandomHorizontalFlip()` | Random horizontal flip with p=0.5 |
| `RandomRotation(10)` | Random rotation up to ±10 degrees |
| `Normalize((0.5,), (0.5,))` | Normalize to [-1, 1] range |

### Regularization Techniques

| Technique | Purpose | Implementation |
|-----------|---------|----------------|
| **Dropout** | Prevents co-adaptation of neurons | `nn.Dropout(0.2)` for MLP, `nn.Dropout2d(0.25)` for CNN |
| **Batch Normalization** | Stabilizes training, allows higher LR | `nn.BatchNorm1d`, `nn.BatchNorm2d` |
| **Weight Decay (L2)** | Penalizes large weights | `weight_decay=1e-4` in AdamW |

### Optimizers

| Optimizer | Learning Rate | Use Case |
|-----------|---------------|----------|
| **Adam** | 0.001 | Baseline experiments |
| **AdamW** | 0.001, wd=1e-4 | Weight decay experiments |
| **SGD** | Configurable | Alternative with momentum |

### Loss Functions

| Loss | Task | PyTorch |
|------|------|---------|
| Cross-Entropy | Classification | `nn.CrossEntropyLoss()` |
| MSE | Reconstruction (AutoEncoder) | `nn.MSELoss()` |

### DataLoader Optimization

```python
NUM_WORKERS = 0 if device.type == 'cpu' else 2  # Parallel data loading
PIN_MEMORY = device.type == 'cuda'              # Faster GPU transfer
BATCH_SIZE = 64                                  # Balanced memory/gradient stability
```

---

## 🧪 Experiments & Results

### Experiment Summary

| # | Model | Optimizer | Regularization | Test Accuracy |
|---|-------|-----------|----------------|---------------|
| 1 | MLP Baseline | Adam (lr=0.001) | Dropout + BN | ~89% |
| 2 | CNN Baseline | Adam (lr=0.001) | Dropout + BN | ~93% |
| 3 | CNN + Weight Decay | AdamW (lr=0.001, wd=1e-4) | Dropout + BN + L2 | **~93.44%** |
| 4 | AutoEncoder | Adam (lr=0.001) | BN | MSE: ~0.0285 |

### Key Findings

1. **CNN vs MLP**: CNN outperforms MLP by ~4% due to spatial feature preservation
2. **Weight Decay Effect**: Reduced train-val gap from 3.07% to 2.75%
3. **Convergence Speed**: Adam reaches 85%+ accuracy within 2 epochs
4. **Common Confusions**: Shirt ↔ T-shirt, Pullover ↔ Coat

---

## 📈 Visualizations Included

- Sample images from dataset
- Class distribution bar chart
- Training/validation loss curves
- Training/validation accuracy curves
- Learning rate schedules
- Confusion matrix with heatmap
- Misclassified samples analysis
- AutoEncoder reconstruction comparison
- CNN first-layer filter visualization
- Feature map activations

---

## 🛠️ Technology Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | Latest | Deep learning framework |
| **torchvision** | Latest | Datasets, transforms |
| **torchinfo** | Latest | Model architecture summary |
| **NumPy** | Latest | Numerical operations |
| **Matplotlib** | Latest | Visualization |
| **Seaborn** | Latest | Confusion matrix heatmap |
| **scikit-learn** | Latest | Metrics, classification report |
| **tqdm** | Latest | Progress bars |

### Environment

- **Python**: 3.x
- **Platform**: Windows/Linux/macOS
- **Hardware**: CPU or CUDA-enabled GPU

---

## 🚀 How to Run

### 1. Clone/Download the Repository

```bash
cd HW2-AmirMohammadBasuti-610304080
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/macOS
```

### 3. Install Dependencies

```bash
pip install torch torchvision torchinfo matplotlib numpy seaborn scikit-learn tqdm
```

### 4. Run the Notebook

Open `HW2-AmirMohammadBasuti-610304080.ipynb` in Jupyter Notebook or VS Code and run all cells.

---

## 📁 Project Structure

```
HW2-AmirMohammadBasuti-610304080/
├── HW2-AmirMohammadBasuti-610304080.ipynb  # Main notebook
├── README.md                                # This file
├── data/                                    # Downloaded dataset (auto-created)
│   └── FashionMNIST/
└── .venv/                                   # Virtual environment (optional)
```

---

## 📝 Notebook Structure

1. **Setup & Imports** - Library imports and device configuration
2. **Data Loading** - Dataset download, augmentation, train/val/test split
3. **Data Exploration** - Sample visualization, class distribution
4. **Model Definitions** - MLP, CNN, AutoEncoderCNN classes
5. **Loss & Optimizer Functions** - Configurable training components
6. **Training Loop** - Generic training function with validation tracking
7. **Experiments** - Four experiments with different configurations
8. **Results Analysis** - Comparison charts, confusion matrix, misclassification analysis
9. **Bonus: Feature Visualization** - CNN filter and feature map visualization
10. **Summary** - Key findings and future work

---

## 🎯 Future Improvements

- [ ] Learning rate schedulers (ReduceLROnPlateau, CosineAnnealing)
- [ ] Expanded augmentation (RandomCrop, ColorJitter, Cutout)
- [ ] Deeper architectures (ResNet-18)
- [ ] Ensemble methods (MLP + CNN voting)
- [ ] Mixed precision training for faster GPU training
- [ ] Early stopping with patience

---

## 📚 References

- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)

---

## 📄 License

This project is submitted as part of coursework for Machine Learning course.

---

*Last Updated: December 2025*

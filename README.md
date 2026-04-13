# 🧠 CNN for CIFAR-10 Classification

A Convolutional Neural Network (CNN) built with **PyTorch** to classify images from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset — achieving **~74.7% test accuracy** after 10 epochs of training.

---

## 📌 Overview

CIFAR-10 is a benchmark image classification dataset containing **60,000 color images** across **10 classes** (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), each of size 32×32 pixels.

This project implements a 3-layer CNN trained end-to-end using cross-entropy loss and the Adam optimizer.

---

## 🏗️ Model Architecture

```
Input (3 × 32 × 32)
    │
    ▼
Conv2d(3 → 32, 3×3, padding=1) → ReLU → MaxPool2d(2×2)
    │
    ▼
Conv2d(32 → 64, 3×3, padding=1) → ReLU → MaxPool2d(2×2)
    │
    ▼
Conv2d(64 → 128, 3×3, padding=1) → ReLU → MaxPool2d(2×2)
    │
    ▼
Flatten (128 × 4 × 4 = 2048)
    │
    ▼
Linear(2048 → 256) → ReLU
    │
    ▼
Linear(256 → 10)
    │
    ▼
Output (10 classes)
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Training Epochs | 10 |
| Final Training Loss | ~0.113 |
| Test Accuracy | **74.71%** |

### Training Loss Curve

| Epoch | Loss |
|-------|------|
| 1 | 0.7556 |
| 2 | 0.6322 |
| 3 | 0.5293 |
| 4 | 0.4282 |
| 5 | 0.3469 |
| 6 | 0.2791 |
| 7 | 0.2118 |
| 8 | 0.1725 |
| 9 | 0.1350 |
| 10 | 0.1135 |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision

### Install Dependencies

```bash
pip install torch torchvision
```

---

## 🚀 Usage

### Clone the Repository

```bash
git clone https://github.com/your-username/cnn-cifar10.git
cd cnn-cifar10
```

### Run the Notebook

Open and run the Jupyter notebook:

```bash
jupyter notebook CNN_for_CIFAR10.ipynb
```

The CIFAR-10 dataset will be **automatically downloaded** to `./data/` on first run.

---

## 📁 Project Structure

```
cnn-cifar10/
│
├── CNN_for_CIFAR10.ipynb   # Main notebook (data loading, model, training, evaluation)
├── data/                   # CIFAR-10 dataset (auto-downloaded)
└── README.md
```

---

## 🔧 Hyperparameters

| Parameter | Value |
|---|---|
| Batch Size | 64 |
| Epochs | 10 |
| Optimizer | Adam (default lr=0.001) |
| Loss Function | CrossEntropyLoss |
| Normalization | mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) |

---

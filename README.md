# Deep Learning Programming — Assignment 1

A from-scratch implementation of a Multi-Layer Perceptron (MLP) in NumPy, verified against PyTorch, and applied to the **Fashion-MNIST** dataset. The project builds up from raw forward/backward propagation to activation and loss function comparisons, optimizer benchmarking, bias–variance diagnosis, regularisation techniques, and a final randomized hyperparameter search with cross-validation.

## Dataset

- **Fashion-MNIST** (via `kagglehub`, `zalando-research/fashionmnist`) — 10-class grayscale clothing image classification.
- 60,000 training images, 10,000 test images, each 28×28 pixels.
- Pixel values normalized to `[0, 1]` and flattened to 784-dimensional vectors.
- Training data split 80/20 (stratified) into **48,000 training** / **12,000 validation** samples; **10,000** held out as the test set.

## Project Structure (Notebook Parts)

| Part | Topic | Summary |
|---|---|---|
| Setup | Data pipeline | Download, extract, reshape, normalize, flatten, and split Fashion-MNIST |
| **Part 1** | MLP from scratch | He-normal weight initialization; 2-layer feed-forward network built with NumPy; forward pass returns output + cache for backprop |
| **Part 2** | PyTorch verification & activations | Cross-checked NumPy loss/gradients against PyTorch (max abs. difference ~1e-8); compared **Sigmoid, Tanh, ReLU, Leaky ReLU** activations, including a dead-ReLU-unit analysis |
| **Part 3** | Loss functions | Compared Cross-Entropy vs. MSE for classification convergence; ran a supplementary tabular regression experiment |
| **Part 4** | Optimizers | Benchmarked **SGD, SGD+Momentum, RMSProp, Adam** at a shared learning rate and again with per-optimizer tuned learning rates |
| **Part 5** | Bias–variance diagnosis | Deliberately overfit a 4-hidden-layer, 512-unit network on 2,000 samples to diagnose high variance (99.05% train vs. 82.79% val accuracy) |
| **Part 6** | Regularisation | Compared **L2 weight decay, L1 penalty, Dropout, Batch Normalisation, Early Stopping, Data Augmentation**, and additional training data as fixes for the Part 5 overfitting |
| **Part 7** | Hyperparameter tuning | Randomized search (12 configurations) with 5-fold cross-validation over learning rate, hidden width, and dropout rate; retrained the best configuration on the full dataset and evaluated on the held-out test set |

## Key Results

### Activation Function Comparison (Part 2)

| Activation | Val Loss | Val Accuracy |
|---|---|---|
| Sigmoid | 0.3112 | 89.03% |
| Tanh | 0.3261 | 88.81% |
| ReLU | 0.3208 | 89.41% |
| Leaky ReLU | 0.3291 | 89.51% |

### Optimizer Comparison (Part 4, tuned learning rates)

| Optimizer | Learning Rate | Val Accuracy |
|---|---|---|
| SGD | 0.01 | 83.96% |
| SGD + Momentum | 0.01 | 88.37% |
| RMSProp | 0.001 | 89.63% |
| **Adam** | 0.001 | 89.41% |

Adam was selected as the preferred optimizer for later parts for reaching strong accuracy quickly with stable convergence.

### Bias–Variance Diagnosis (Part 5)

Training on 2,000 samples with an oversized network (4×512 hidden units) produced a clear **high-variance** signature:

- Training accuracy: **99.05%**
- Validation accuracy: **82.79%**
- Generalisation gap: **16.26%**

### Regularisation Comparison (Part 6)

| Technique | Setting | Train Acc | Val Acc | Gap |
|---|---|---|---|---|
| Baseline | — | 99.05% | 82.79% | 16.26% |
| L2 weight decay | λ=0.001 | 98.40% | 82.81% | 15.59% |
| Dropout | rate=0.6 | 90.45% | 82.28% | 8.17% |
| Batch Normalisation | — | 99.95% | 82.17% | 17.78% |
| **Early Stopping** | patience=3 | 88.90% | 81.48% | **7.42%** |
| Data Augmentation | flip + rotation ±10° | 92.70% | 82.41% | 10.29% |
| More training data | 20,000 samples | 97.56% | 87.71% | 9.85% |

Early stopping produced the largest reduction in generalisation gap relative to training-accuracy cost.

### Final Hyperparameter Search & Evaluation (Part 7)

- Randomized search over 12 configurations, evaluated with 5-fold cross-validation.
- **Selected configuration:** learning rate ≈ 0.000155, hidden width = 768, dropout = 0.2 (mean CV accuracy 89.36%, std 0.64%).
- Retrained on the full 60,000-sample dataset for 15 epochs (25.01s).

**Final test set performance:**

| Metric | Score |
|---|---|
| Test Accuracy | 88.97% |
| Macro Precision | 89.30% |
| Macro Recall | 88.97% |
| Macro F1 | 88.96% |

Compared to the Part 2 ReLU baseline (89.41% validation accuracy), the tuned model scored 0.44 percentage points lower on the held-out test set — a reminder that cross-validation selects for average fold performance, which can diverge slightly from final held-out test performance.

## Tools & Libraries

`numpy` · `pandas` · `scikit-learn` · `matplotlib` · `torch` · `kagglehub`

## How to Run

1. Install dependencies: `pip install numpy pandas scikit-learn matplotlib torch kagglehub`
2. Open the notebook and run all cells in order — the dataset is downloaded automatically via `kagglehub`.
3. GPU (CUDA) is used automatically where available for the PyTorch verification steps; the core MLP implementation runs on NumPy/CPU.

## Author

Ahmad — Roll No. 23F-0707
NUCES Chiniot-Faisalabad Campus, AL2002 (Artificial Intelligence)

# Sentiment Analysis: From Scratch ML Pipeline

> Building and benchmarking NLP sentiment classifiers from the ground up — implementing gradient descent, comparing linear models, and analyzing scalability on real-world text data.

---

## Project Overview

This project builds a complete **sentiment analysis pipeline** on the Stanford Sentiment Treebank dataset, classifying movie reviews as positive (+1) or negative (-1). Rather than relying on black-box libraries, the focus is on **understanding what's happening under the hood** — implementing optimization algorithms from scratch and rigorously comparing model performance, speed, and scalability.

---

## Key Highlights

- **Custom NLP pipeline** — tokenization and Bag-of-Words featurization producing 13,297-dimensional feature vectors
- **Gradient Descent from scratch** — full implementation in NumPy with learning rate experimentation
- **SGD from scratch** — mini-batch Stochastic Gradient Descent with configurable batch size and epochs
- **Model benchmarking** — systematic comparison of 6 models across accuracy, speed, and scalability
- **Scalability analysis** — identifying which models survive training on 40,000+ examples within 2 minutes

---

## Project Structure

```
sentiment-analysis/
│
├── Sentiment-Analysis.ipynb   # Main notebook with full pipeline
├── data.tsv                   # Stanford Sentiment Treebank dataset
└── README.md
```

---

## NLP Pipeline

Raw text is transformed into machine-readable feature vectors through a custom pipeline:

1. **Tokenization** — lowercasing and regex-based word extraction (handles apostrophes e.g. *don't*)
2. **Vocabulary construction** — built from training data only (13,297 unique words)
3. **Bag-of-Words encoding** — each sentence becomes a sparse vector of word counts
4. **Train / Val / Test split** — 60% / 20% / 20% stratified split for unbiased evaluation

```python
def tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
```

---

## Models Compared

| Model | Test Accuracy | Training Time (5k samples) |
|---|---|---|
| Linear Regression | 68% | 113s |
| Ridge Regression (α=10) | 78% | 16s |
| Logistic Regression | 79% | 10s |
| LinearSVC | 79% | 2s |
| Gradient Descent (custom) | 78% | ~220s |
| SGD (custom) | 79% | ~9s |

**Baseline (majority class):** 55.5%

---

## From-Scratch Implementations

### Gradient Descent
Full batch gradient descent with configurable learning rate and iterations:

```python
grad_w = X.T.dot(error) / len(error)
grad_b = np.mean(error)
self.w -= self.lr * grad_w
self.b -= self.lr * grad_b
```

**Learning rate experiments** (10ᵏ for k ∈ {-2, -1, 0, 1, 2}):
- `lr = 0.01, 0.1` → too slow, model hasn't converged after 500 iterations
- `lr = 1.0` → optimal, 78% test accuracy
- `lr = 10, 100` → diverges, loss explodes to infinity

### Mini-Batch SGD
Stochastic Gradient Descent with shuffling and configurable batch size:

- **20-40x faster** than full batch GD
- Achieves **79% test accuracy** in ~9 seconds vs 220+ seconds for full batch GD
- Smaller batch sizes (10-15) converge faster per epoch due to more frequent weight updates

---

## 📈 Scalability Analysis

Tested on full training set (40,409 examples × 13,297 features):

| Model | Scalable? | Reason |
|---|---|---|
| Linear Regression | ❌ | O(d³) matrix computation |
| Ridge Regression | ❌ | RAM overflow on full data |
| Logistic Regression | ❌ | RAM overflow on full data |
| Full Batch GD | ❌ | ~30 min estimated |
| **LinearSVC** | ✅ | **90% test accuracy in 7s** |
| **SGD (custom)** | ✅ | **87% test accuracy in ~18s** |

**Best result: 90% test accuracy with LinearSVC on full dataset** — a 11 point improvement over the 5,000 sample subset, demonstrating the direct impact of training data volume on model performance.

---

## Key Takeaways

- **Regularization matters** — Ridge regression reduced overfitting from 68% → 78% test accuracy by penalizing large weights
- **Loss function design is critical** — logistic loss and hinge loss outperform MSE for classification because they're explicitly designed for binary outputs
- **SGD scales, full batch GD doesn't** — processing mini-batches makes 10,000 weight updates in the time full batch GD makes 500
- **More data = better models** — SVM jumped from 79% → 90% accuracy when trained on 8x more data

---

## Tech Stack

- Python 3
- NumPy
- scikit-learn
- pandas

---

## Getting Started

```bash
git clone https://github.com/GodblessOsei/sentiment-from-scratch
cd sentiment-analysis
pip install numpy pandas scikit-learn
# Add data.tsv to the directory
jupyter notebook Sentiment-Analysis.ipynb
```

---

## Dataset

Stanford Sentiment Treebank (SST) — a benchmark dataset of movie review sentences with human-annotated sentiment labels.

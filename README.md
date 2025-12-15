# Tabular Classification Benchmark

This repository implements a **reproducible, code-first benchmarking pipeline** for classical machine learning models on a real-world tabular dataset.

The goal of this project is to demonstrate **end-to-end ML engineering fundamentals**. The focus is on data preprocessing, model comparison, evaluation metrics, and clean software structure.

---

## Problem Statement

Given a tabular dataset with mixed numerical and categorical features, the task is to predict a **binary target variable** (`income`) using multiple classical ML algorithms and compare their performance under the same preprocessing and evaluation conditions.

---

## Dataset

**UCI Adult Income Dataset**

- Type: Binary classification
- Samples: ~48,000
- Features:
  - Numerical (age, hours per week, capital gain/loss)
  - Categorical (education, occupation, marital status, etc.)
- Target:
  - `income`
    - `0`: ≤ 50K
    - `1`: > 50K

This dataset was chosen because:

- It reflects real-world data issues (categorical features, imbalance)
- It is commonly used in industry interviews
- It requires non-trivial preprocessing

---

## Models Benchmarked

The following models are trained and evaluated using a shared pipeline:

- **Logistic Regression**

  - Strong baseline
  - Interpretable
  - Fast training and inference

- **Support Vector Machine (SVM)**

  - Non-linear decision boundary
  - Higher computational cost
  - Often strong on structured data

- **Random Forest**

  - Ensemble of decision trees
  - Handles non-linear relationships well
  - Robust to feature interactions

- **XGBoost**
  - Gradient-boosted decision trees
  - Typically state-of-the-art for tabular data
  - Strong bias-variance tradeoff

---

## Project Structure

```
tabular-classification-benchmark/
├── data/
│ └── data.csv # Preprocessed dataset
├── src/
│ ├── data.py # Data loading + preprocessing
│ ├── models.py # Model factory
│ ├── train.py # Training entry point
│ ├── evaluate.py # Metrics computation
│ └── utils.py # Helper utilities
├── results/
│ └── metrics.json # Stored evaluation results
├── requirements.txt
└── README.md
```

---

The project is intentionally **not notebook-driven**. Training and evaluation logic is implemented as Python modules to reflect real-world ML workflows.

---

## Data Preprocessing

The preprocessing pipeline includes:

1. Loading raw CSV data
2. Separating features and target
3. One-hot encoding categorical variables
4. Train/test split with stratification
5. Feature scaling using `StandardScaler`

All models are trained on the **same transformed data** to ensure a fair comparison.

---

## Evaluation Metrics

Each model is evaluated on the held-out test set using the following metrics:

- Accuracy
- ROC-AUC
- F1 Score
- Precision
- Recall
- Training Time (seconds)

ROC-AUC and F1 are emphasized over accuracy due to potential class imbalance.

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train and evaluate a model:

```
python src/train.py --model logistic
python src/train.py --model svm
python src/train.py --model rf
python src/train.py --model xgb

```

## Evaluation results are saved to:

```
results/metrics.json
```

# ENSIA Amphitheatre GPS Localization — HMM Implementation Guide

> **For GitHub Copilot / AI coding assistants:**  
> This file is the authoritative specification for the HMM-based GPS localization system.
> Each section is structured as an implementation task. Follow the outlined types, shapes,
> and algorithms exactly. Use `hmmlearn`, `numpy`, and `scikit-learn` unless noted otherwise.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Schema](#2-dataset-schema)
3. [HMM Problem Formulation](#3-hmm-problem-formulation)
4. [Implementation Tasks](#4-implementation-tasks)
   - 4.1 [Data loading & preprocessing](#41-data-loading--preprocessing)
   - 4.2 [Sequence construction](#42-sequence-construction)
   - 4.3 [Model definition](#43-model-definition)
   - 4.4 [Training — Baum–Welch](#44-training--baumwelch)
   - 4.5 [Decoding — Viterbi](#45-decoding--viterbi)
   - 4.6 [Evaluation](#46-evaluation)
5. [File Structure](#5-file-structure)
6. [Dependencies](#6-dependencies)
7. [Implementation Notes & Edge Cases](#7-implementation-notes--edge-cases)

---

## 1. Project Overview

Students at ENSIA submit GPS readings while attending lectures in one of several amphitheatres
(Amphi 1 – Amphi 7) or from outside. Each submission contains a sequence of raw GPS pings
collected over a short window.

**Goal:** given only the GPS signal sequence, predict the correct amphitheatre label (or Outside).

**Why HMM:**  
- The true location is a *hidden state* — GPS does not directly reveal which room you are in.  
- GPS readings are *sequential observations* emitted by the hidden state.  
- The state is *persistent* within a session: a student stays in one amphitheatre.

---

## 2. Dataset Schema

**File:** `Data/ensia_gps_data.csv`

| Column | Type | Description |
|---|---|---|
| `Amphi` | `str` | Target label — `"Amphi 1"` … `"Amphi 7"` or `"Outside"` |
| `Lat_Mean` | `float` | Mean latitude across raw readings |
| `Lng_Mean` | `float` | Mean longitude across raw readings |
| `Acc_Mean` | `float` | Mean GPS accuracy (metres; lower = better) |
| `RawReadings` | `str` (JSON) | Array of `{lat, lng, accuracy, timestamp}` objects |
| `IsOutside` | `bool` | True if the student is outdoors |
| `Year` | `int` | Student academic year |
| `Module` | `str` | Course name |
| `Block`, `Row`, `Column` | `int` | Seat position metadata |

### Parsing `RawReadings`

```python
import json, pandas as pd

df = pd.read_csv("Data/ensia_gps_data.csv")
df["RawReadings"] = df["RawReadings"].apply(json.loads)
# Each element: {"lat": float, "lng": float, "accuracy": float, "timestamp": int}
```

---

## 3. HMM Problem Formulation

### 3.1 State space S

```
S = { "Amphi 1", "Amphi 2", "Amphi 3", "Amphi 4",
      "Amphi 5", "Amphi 6", "Amphi 7", "Outside" }
|S| = 8
```

Map to integer indices 0–7. Keep a `label_encoder` (e.g. `sklearn.LabelEncoder`) for
round-tripping between string labels and integer state indices.

### 3.2 Observation space O

Each observation at time step `t` is a 3-dimensional continuous vector:

```
o_t = [lat_t, lng_t, acc_t]  ∈ ℝ³
```

A full session of T pings produces the observation sequence:

```
O = [o_1, o_2, ..., o_T],  shape: (T, 3)
```

### 3.3 Transition matrix A

```
A[i][j] = P(s_{t+1} = j | s_t = i)
Shape: (|S|, |S|) = (8, 8)
Row-stochastic: each row sums to 1.
```

**Expected structure:**  
Within a session the student does not change rooms, so the diagonal should dominate.
Initialise with a strong diagonal prior and let Baum–Welch refine it.

```python
# Suggested initialisation
import numpy as np
A_init = np.eye(n_states) * 0.9 + np.ones((n_states, n_states)) * 0.1 / n_states
A_init /= A_init.sum(axis=1, keepdims=True)
```

### 3.4 Emission model B

Because observations are continuous, model each state as a **multivariate Gaussian**:

```
B(o_t | s_i) = N(o_t ; μ_i, Σ_i)

μ_i ∈ ℝ³        — mean GPS vector for state i
Σ_i ∈ ℝ³ˣ³      — covariance (full or diagonal)
```

Use `hmmlearn.hmm.GaussianHMM` with `covariance_type="full"` (or `"diag"` if data is scarce).

**Physical intuition:**
- Indoor amphitheatres: tight lat/lng cluster, elevated `acc` value (poor GPS).
- Outside: wider lat/lng spread, low `acc` value (strong GPS signal).

### 3.5 Initial state distribution π

```
π[i] = P(s_1 = s_i)
Shape: (|S|,)  — sums to 1
```

Estimate from training label frequencies:

```python
from collections import Counter
counts = Counter(train_labels)
pi = np.array([counts.get(i, 0) for i in range(n_states)], dtype=float)
pi /= pi.sum()
```

### 3.6 The three HMM problems

| Problem | What it answers | Algorithm | Use in this project |
|---|---|---|---|
| **Evaluation** | P(O \| λ) | Forward algorithm | Score sequences against per-class models |
| **Decoding** | argmax P(S \| O, λ) | Viterbi | Predict amphitheatre from a new GPS sequence |
| **Learning** | argmax λ P(O \| λ) | Baum–Welch (EM) | Fit A, B, π to training sequences |

---

## 4. Implementation Tasks

### 4.1 Data loading & preprocessing

**File:** `Preprocessing/preprocess.py`

```python
# TODO: implement load_and_clean()
# Steps:
#   1. Read CSV with pd.read_csv()
#   2. Parse RawReadings JSON column
#   3. Drop rows where RawReadings is empty or has < 2 pings
#   4. Encode Amphi labels to integers with LabelEncoder
#   5. Normalise lat, lng, acc with StandardScaler
#      — fit scaler on TRAIN set only, transform both train and test
#   6. Return (df_train, df_test, label_encoder, scaler)

def load_and_clean(csv_path: str, test_size: float = 0.2):
    ...
```

**Normalisation is critical.** Latitude differences of 0.001° ≈ 111 m, while accuracy spans
0–50+ metres. Without scaling the Gaussian covariances will be ill-conditioned.

### 4.2 Sequence construction

**File:** `Modeling/sequences.py`

```python
# TODO: implement build_sequences()
# Each row in df becomes one observation sequence (shape: T_i x 3)
# and one label (the integer-encoded Amphi).
#
# Returns:
#   sequences : list of np.ndarray, each shape (T_i, 3)
#   lengths   : list of int  [T_1, T_2, ..., T_N]  — required by hmmlearn
#   labels    : np.ndarray of int, shape (N,)

def build_sequences(df: pd.DataFrame) -> tuple[list, list, np.ndarray]:
    sequences, lengths, labels = [], [], []
    for _, row in df.iterrows():
        pings = row["RawReadings"]
        obs = np.array([[p["lat"], p["lng"], p["accuracy"]] for p in pings])
        sequences.append(obs)
        lengths.append(len(obs))
        labels.append(row["label"])  # integer-encoded
    return sequences, lengths, np.array(labels)
```

**hmmlearn input format:** concatenate all sequences into one array and pass `lengths`:

```python
X_concat = np.concatenate(sequences, axis=0)  # shape: (sum(lengths), 3)
```

### 4.3 Model definition

**File:** `Modeling/hmm_model.py`

```python
from hmmlearn.hmm import GaussianHMM

# TODO: implement build_model()
# One GaussianHMM per class (one-vs-rest generative approach)
# OR a single model trained jointly (see Note below).

def build_model(n_states: int = 1, covariance_type: str = "full", n_iter: int = 100):
    return GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        init_params="stmc",   # let hmmlearn initialise s, t, m, c
        params="stmc",        # learn all parameters
        random_state=42,
    )
```

**Two architectural choices — pick one and document your choice:**

**Option A — Per-class models (recommended for this dataset):**  
Train one `GaussianHMM(n_components=1)` per amphitheatre label.
At inference, score the test sequence against all 8 models with `model.score(O)` (log-likelihood)
and predict the class with the highest score.

```
predict(O) = argmax_i  model_i.score(O)
```

**Option B — Single multi-state model:**  
Train one `GaussianHMM(n_components=8)` on all sequences.
Use Viterbi (`model.predict(O)`) to decode the state sequence, then take the mode state as the prediction.
Requires careful state-to-label alignment after training.

### 4.4 Training — Baum–Welch

**File:** `Modeling/train.py`

```python
# TODO: implement train_per_class_models()
# For each class label i:
#   1. Filter training sequences where labels == i
#   2. Concatenate them: X_i = np.concatenate(seqs_i), lengths_i = [len(s) for s in seqs_i]
#   3. Fit model: model_i.fit(X_i, lengths_i)
#   4. Store model_i in a dict keyed by class index
# Return: dict[int, GaussianHMM]

def train_per_class_models(sequences, lengths, labels, n_iter=100):
    models = {}
    unique_classes = np.unique(labels)
    for c in unique_classes:
        mask = labels == c
        seqs_c = [sequences[i] for i in range(len(sequences)) if mask[i]]
        X_c = np.concatenate(seqs_c)
        lens_c = [len(s) for s in seqs_c]
        model = build_model(n_states=1, n_iter=n_iter)
        model.fit(X_c, lens_c)
        models[c] = model
    return models
```

### 4.5 Decoding — Viterbi

**File:** `Modeling/predict.py`

```python
# TODO: implement predict_sequence()
# Score one observation sequence against all per-class models.
# Return the predicted class label (integer).

def predict_sequence(obs: np.ndarray, models: dict) -> int:
    # obs shape: (T, 3)
    scores = {c: model.score(obs) for c, model in models.items()}
    return max(scores, key=scores.get)

# TODO: implement predict_batch()
# Apply predict_sequence to every sequence in the test set.
# Return np.ndarray of predicted labels.

def predict_batch(sequences: list, models: dict) -> np.ndarray:
    return np.array([predict_sequence(obs, models) for obs in sequences])
```

### 4.6 Evaluation

**File:** `Modeling/evaluate.py`

```python
# TODO: implement evaluate()
# Metrics to compute and print:
#   - Accuracy
#   - Per-class precision, recall, F1 (sklearn.metrics.classification_report)
#   - Confusion matrix (plot with seaborn.heatmap)
# Save confusion matrix figure to figures/confusion_matrix.png

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt

def evaluate(y_true, y_pred, label_encoder, save_path="figures/confusion_matrix.png"):
    labels = label_encoder.classes_
    print(classification_report(y_true, y_pred, target_names=labels))
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to {save_path}")
```

---

## 5. File Structure

```
Project/
├── Data/
│   └── ensia_gps_data.csv
├── Preprocessing/
│   ├── preprocess.py          ← Task 4.1
│   ├── 1-exploration.ipynb
│   └── 2-visualization.ipynb
├── Modeling/
│   ├── sequences.py           ← Task 4.2
│   ├── hmm_model.py           ← Task 4.3
│   ├── train.py               ← Task 4.4
│   ├── predict.py             ← Task 4.5
│   ├── evaluate.py            ← Task 4.6
│   └── brahimi/
│       └── baseline.ipynb
├── figures/
│   └── confusion_matrix.png
├── main.py                    ← Entry point: wire all tasks together
└── README_HMM.md              ← This file
```

### `main.py` skeleton

```python
# TODO: wire the full pipeline
from Preprocessing.preprocess import load_and_clean
from Modeling.sequences import build_sequences
from Modeling.train import train_per_class_models
from Modeling.predict import predict_batch
from Modeling.evaluate import evaluate

df_train, df_test, le, scaler = load_and_clean("Data/ensia_gps_data.csv")
train_seqs, train_lens, train_labels = build_sequences(df_train)
test_seqs,  test_lens,  test_labels  = build_sequences(df_test)

models = train_per_class_models(train_seqs, train_lens, train_labels)
y_pred = predict_batch(test_seqs, models)
evaluate(test_labels, y_pred, le)
```

---

## 6. Dependencies

```txt
pandas>=2.0
numpy>=1.25
scikit-learn>=1.4
hmmlearn>=0.3.0
matplotlib>=3.8
seaborn>=0.13
```

Install with:

```bash
pip install pandas numpy scikit-learn hmmlearn matplotlib seaborn
```

---

## 7. Implementation Notes & Edge Cases

### Convergence warnings
`hmmlearn` may emit `ConvergenceWarning` for classes with very few sequences.
Increase `n_iter` or reduce `covariance_type` to `"diag"` for sparse classes.

### Short sequences (T = 1)
A single-ping session is valid but gives no transition information.
Handle it: `model.score(obs)` still works for T=1 — Baum–Welch just learns the emission.
Flag sessions with T < 3 and consider excluding them from training.

### GPS accuracy scale
`accuracy` in the raw readings is in metres. It is *inversely* related to quality
(accuracy=5 means ±5 m, which is good; accuracy=50 is bad).
After `StandardScaler`, higher (worse) accuracy will have a positive scaled value.
This is intentional — indoor states will cluster at high scaled accuracy.

### Label imbalance
If some amphitheatres have far fewer sessions than others, the per-class models will
be trained on less data and score less reliably. Mitigate by:
- Augmenting short sequences (duplicate with small Gaussian noise added).
- Using a diagonal covariance to reduce parameter count.

### State ordering in Option B (multi-state model)
After training a single 8-state HMM, states are not automatically aligned to class labels.
Align by computing the most common ground-truth label for sequences whose Viterbi-decoded
dominant state is state i:
```python
from scipy.stats import mode
state_to_label = {}
for state in range(n_states):
    mask = (dominant_states == state)
    state_to_label[state] = mode(train_labels[mask]).mode[0]
```

### Reproducibility
Set `random_state=42` in all `GaussianHMM` constructors and `train_test_split`.

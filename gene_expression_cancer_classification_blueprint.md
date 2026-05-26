# 🧬 Gene Expression Cancer Classification
## Full-Stack AI Project Blueprint — Codex Agent Specification

---

## 📋 PROJECT OVERVIEW

**Goal:** Build an end-to-end machine learning web application that classifies cancer types from gene expression (RNA-Seq) data using deep neural networks, with a clean interactive frontend for inference and visualization.

**Dataset:** TCGA (The Cancer Genome Atlas) — Gene Expression by RNA-Seq
- Source: https://www.cancer.gov/tcga or https://xenabrowser.net/datapages/
- Fallback (simpler): UCI ML Repo — Gene Expression Cancer RNA-Seq Data Set
  - Direct: https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq
- Samples: ~800 patients × 20,531 gene features
- Classes: BRCA, KIRC, COAD, LUAD, PRAD (5 cancer types)

---

## 🗂️ PROJECT STRUCTURE

```
gene-cancer-classifier/
│
├── data/
│   ├── raw/                        # Downloaded TCGA/UCI dataset CSV files
│   │   ├── data.csv                # Gene expression matrix (samples × genes)
│   │   └── labels.csv              # Cancer type labels per sample
│   ├── processed/                  # Preprocessed numpy arrays
│   │   ├── X_train.npy
│   │   ├── X_test.npy
│   │   ├── y_train.npy
│   │   └── y_test.npy
│   └── README.md                   # Dataset download instructions
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Data cleaning & feature selection
│   └── 03_model_experiments.ipynb  # Model comparison experiments
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py             # Auto-download dataset script
│   │   ├── preprocess.py           # Preprocessing pipeline
│   │   └── dataset.py              # PyTorch Dataset class
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py             # Sklearn baseline models (RF, SVM, XGBoost)
│   │   ├── mlp.py                  # Multi-Layer Perceptron (PyTorch)
│   │   ├── deep_net.py             # Deep Neural Network with BatchNorm + Dropout
│   │   └── autoencoder.py          # Optional: Autoencoder for dimensionality reduction
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop with early stopping
│   │   ├── callbacks.py            # Checkpointing, logging callbacks
│   │   └── evaluate.py             # Metrics, confusion matrix, ROC curves
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Hyperparameters & paths config
│       ├── logger.py               # Logging setup
│       └── visualization.py        # Plot helpers
│
├── app/
│   ├── main.py                     # FastAPI backend
│   ├── schemas.py                  # Pydantic request/response models
│   ├── inference.py                # Model loading & prediction logic
│   └── frontend/
│       ├── index.html              # Main UI page
│       ├── style.css               # Styling
│       └── app.js                  # Frontend JavaScript
│
├── models/
│   ├── saved/                      # Serialized trained models (.pth, .pkl)
│   │   ├── deep_net_best.pth
│   │   ├── random_forest.pkl
│   │   └── label_encoder.pkl
│   └── artifacts/
│       ├── scaler.pkl              # Fitted StandardScaler
│       └── feature_names.json      # Selected gene feature names
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
│
├── scripts/
│   ├── train.py                    # CLI training script
│   ├── evaluate.py                 # CLI evaluation script
│   └── predict.py                  # CLI single-sample prediction
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## ⚙️ TECH STACK

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | All backend & ML code |
| Deep Learning | PyTorch 2.x | Neural network training & inference |
| ML Baselines | Scikit-learn, XGBoost | Comparison models |
| Data Processing | Pandas, NumPy | Tabular data manipulation |
| Visualization | Matplotlib, Seaborn, Plotly | Plots and charts |
| Dimensionality Reduction | sklearn PCA / UMAP | Feature reduction for viz |
| API Backend | FastAPI | REST API serving predictions |
| Frontend | HTML + CSS + Vanilla JS | Interactive prediction UI |
| Model Serialization | torch.save / joblib | Saving/loading models |
| Environment | python-dotenv | Config management |
| Testing | pytest | Unit & integration tests |
| Containerization | Docker + Docker Compose | Reproducible deployment |

---

## 📦 REQUIREMENTS FILE

```txt
# requirements.txt

# Core ML
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
xgboost>=1.7.0
numpy>=1.24.0
pandas>=2.0.0

# Data & Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
umap-learn>=0.5.3

# API & Web
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
pydantic>=2.0.0
python-multipart>=0.0.6

# Utilities
joblib>=1.3.0
python-dotenv>=1.0.0
tqdm>=4.65.0
requests>=2.31.0

# Notebooks
jupyter>=1.0.0
ipykernel>=6.23.0

# Testing
pytest>=7.4.0
httpx>=0.24.0
```

---

## 🔧 CONFIGURATION — `src/utils/config.py`

```python
# src/utils/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Paths
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = BASE_DIR / "models" / "saved"
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"

# Dataset
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz"
CANCER_CLASSES = ["BRCA", "KIRC", "COAD", "LUAD", "PRAD"]
NUM_CLASSES = 5
INPUT_FEATURES = 20531   # raw; reduced after feature selection

# Preprocessing
TOP_K_FEATURES     = 2000   # Select top-K genes by variance
TEST_SIZE          = 0.2
RANDOM_STATE       = 42
NORMALIZE          = True   # StandardScaler

# Model — Deep Neural Network
HIDDEN_LAYERS      = [1024, 512, 256, 128]
DROPOUT_RATE       = 0.4
BATCH_NORM         = True
ACTIVATION         = "relu"

# Training
BATCH_SIZE         = 32
LEARNING_RATE      = 1e-3
WEIGHT_DECAY       = 1e-4
EPOCHS             = 100
EARLY_STOPPING     = 10     # patience in epochs
SCHEDULER          = "cosine"

# API
API_HOST           = "0.0.0.0"
API_PORT           = 8000
MODEL_PATH         = MODELS_DIR / "deep_net_best.pth"
SCALER_PATH        = ARTIFACTS_DIR / "scaler.pkl"
ENCODER_PATH       = ARTIFACTS_DIR / "label_encoder.pkl"
FEATURES_PATH      = ARTIFACTS_DIR / "feature_names.json"
```

---

## 📥 DATA PIPELINE — `src/data/preprocess.py`

```python
# src/data/preprocess.py
"""
Full preprocessing pipeline:
1. Load raw CSV data
2. Handle missing values
3. Select top-K high-variance genes
4. Split train/test
5. Normalize with StandardScaler
6. Encode labels
7. Save processed arrays to /data/processed/
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

from utils.config import (
    RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR,
    TOP_K_FEATURES, TEST_SIZE, RANDOM_STATE
)

def load_raw_data():
    """Load expression matrix and labels."""
    X = pd.read_csv(RAW_DIR / "data.csv", index_col=0)
    y = pd.read_csv(RAW_DIR / "labels.csv", index_col=0).squeeze()
    return X, y

def select_features(X: pd.DataFrame, k: int = TOP_K_FEATURES) -> pd.DataFrame:
    """Select top-K genes by variance."""
    variances = X.var(axis=0)
    top_genes = variances.nlargest(k).index
    return X[top_genes]

def preprocess(save: bool = True):
    X, y = load_raw_data()

    # Remove zero-variance genes
    selector = VarianceThreshold()
    X_filtered = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])

    # Feature selection
    X_selected = select_features(X_filtered)
    feature_names = X_selected.columns.tolist()

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected.values, y_encoded,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(PROCESSED_DIR / "X_train.npy", X_train)
        np.save(PROCESSED_DIR / "X_test.npy",  X_test)
        np.save(PROCESSED_DIR / "y_train.npy", y_train)
        np.save(PROCESSED_DIR / "y_test.npy",  y_test)
        joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
        joblib.dump(le,     ARTIFACTS_DIR / "label_encoder.pkl")
        with open(ARTIFACTS_DIR / "feature_names.json", "w") as f:
            json.dump(feature_names, f)
        print(f"✅ Saved processed data. Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler, le, feature_names
```

---

## 🧠 DEEP NEURAL NETWORK MODEL — `src/models/deep_net.py`

```python
# src/models/deep_net.py
import torch
import torch.nn as nn
from utils.config import (
    TOP_K_FEATURES, NUM_CLASSES,
    HIDDEN_LAYERS, DROPOUT_RATE, BATCH_NORM
)

class CancerClassifierDNN(nn.Module):
    """
    Deep Neural Network for gene expression cancer classification.
    Architecture: Input → [Linear → BatchNorm → ReLU → Dropout] × N → Output
    """

    def __init__(
        self,
        input_dim: int = TOP_K_FEATURES,
        hidden_layers: list = HIDDEN_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = DROPOUT_RATE,
        use_batch_norm: bool = BATCH_NORM,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
```

---

## 🏋️ TRAINING LOOP — `src/training/trainer.py`

```python
# src/training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path

from utils.config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EPOCHS, EARLY_STOPPING, MODELS_DIR
)

class Trainer:
    def __init__(self, model, device=None):
        self.model  = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _make_loaders(self, X_train, y_train, X_val, y_val):
        def to_tensor(X, y):
            return TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long)
            )
        train_ds = to_tensor(X_train, y_train)
        val_ds   = to_tensor(X_val,   y_val)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
        return train_loader, val_loader

    def train(self, X_train, y_train, X_val, y_val):
        train_loader, val_loader = self._make_loaders(X_train, y_train, X_val, y_val)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_acc  = 0
        patience_ctr  = 0
        history       = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, EPOCHS + 1):
            # --- Train ---
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            # --- Validate ---
            val_loss, val_acc = self._evaluate(val_loader, criterion)
            scheduler.step()

            history["train_loss"].append(train_loss / len(train_loader))
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss/len(train_loader):.4f} "
                  f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # --- Early Stopping & Checkpointing ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_ctr = 0
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), MODELS_DIR / "deep_net_best.pth")
                print(f"  ✅ New best model saved (val_acc={val_acc:.4f})")
            else:
                patience_ctr += 1
                if patience_ctr >= EARLY_STOPPING:
                    print(f"⏹ Early stopping at epoch {epoch}")
                    break

        return history

    def _evaluate(self, loader, criterion):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                total_loss += criterion(logits, y_batch).item()
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total   += y_batch.size(0)
        return total_loss / len(loader), correct / total
```

---

## 📊 EVALUATION — `src/training/evaluate.py`

```python
# src/training/evaluate.py
"""
Generates:
- Classification report (precision, recall, F1 per class)
- Confusion matrix heatmap
- ROC-AUC curves (one-vs-rest)
- PCA and UMAP embeddings visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

# ---- Confusion Matrix ----
def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix — Gene Expression Cancer Classifier")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

# ---- ROC Curves ----
def plot_roc_curves(y_true, y_proba, class_names, save_path=None):
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    plt.figure(figsize=(9, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — One-vs-Rest")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

# ---- PCA Embedding ----
def plot_pca_embedding(X, y, class_names, save_path=None):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        mask = y == i
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=name, alpha=0.6, s=20)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA of Gene Expression Features")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

# ---- Full Report ----
def full_evaluation(model, X_test, y_test, class_names, device):
    import torch
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        proba = model.predict_proba(X_t).cpu().numpy()
    y_pred = proba.argmax(axis=1)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_roc_curves(y_test, proba, class_names)
    plot_pca_embedding(X_test, y_test, class_names)
    return y_pred, proba
```

---

## 🌐 FASTAPI BACKEND — `app/main.py`

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

from app.inference import CancerInferenceEngine

app = FastAPI(
    title="🧬 Gene Expression Cancer Classifier API",
    description="Predict cancer type from RNA-Seq gene expression data",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/frontend", StaticFiles(directory="app/frontend"), name="frontend")

engine = CancerInferenceEngine()

class ExpressionInput(BaseModel):
    gene_values: List[float]          # Raw expression values (must match feature_names length)
    sample_id: str = "sample_001"

class PredictionResponse(BaseModel):
    sample_id: str
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    top_features: List[dict]          # Top contributing genes via feature importance

@app.get("/")
def root():
    return {"message": "Gene Expression Cancer Classifier API", "docs": "/docs"}

@app.get("/classes")
def get_classes():
    return {"classes": engine.class_names}

@app.get("/features")
def get_features():
    return {"feature_names": engine.feature_names, "count": len(engine.feature_names)}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ExpressionInput):
    if len(payload.gene_values) != len(engine.feature_names):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(engine.feature_names)} features, got {len(payload.gene_values)}"
        )
    result = engine.predict(payload.gene_values)
    return PredictionResponse(sample_id=payload.sample_id, **result)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": engine.model_loaded}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 🔮 INFERENCE ENGINE — `app/inference.py`

```python
# app/inference.py
import torch
import joblib
import json
import numpy as np
from pathlib import Path
from src.models.deep_net import CancerClassifierDNN
from src.utils.config import (
    MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURES_PATH,
    TOP_K_FEATURES, HIDDEN_LAYERS, DROPOUT_RATE, NUM_CLASSES
)

class CancerInferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        self.scaler      = joblib.load(SCALER_PATH)
        self.le          = joblib.load(ENCODER_PATH)
        self.class_names = list(self.le.classes_)
        with open(FEATURES_PATH) as f:
            self.feature_names = json.load(f)

        self.model = CancerClassifierDNN(
            input_dim=len(self.feature_names),
            hidden_layers=HIDDEN_LAYERS,
            num_classes=NUM_CLASSES,
            dropout_rate=DROPOUT_RATE,
        )
        state = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.model_loaded = True

    def predict(self, gene_values: list) -> dict:
        x = np.array(gene_values, dtype=np.float32).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            proba = self.model.predict_proba(x_tensor).cpu().numpy()[0]

        pred_idx   = int(np.argmax(proba))
        pred_class = self.class_names[pred_idx]
        confidence = float(proba[pred_idx])

        class_probabilities = {
            cls: round(float(p), 4)
            for cls, p in zip(self.class_names, proba)
        }

        # Top 5 highest-expressed genes in this sample
        top_indices  = np.argsort(x_scaled[0])[::-1][:5]
        top_features = [
            {"gene": self.feature_names[i], "expression": round(float(x_scaled[0][i]), 4)}
            for i in top_indices
        ]

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "class_probabilities": class_probabilities,
            "top_features": top_features,
        }
```

---

## 🖥️ FRONTEND — `app/frontend/index.html`

```html
<!-- app/frontend/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>🧬 Cancer Gene Classifier</title>
  <link rel="stylesheet" href="style.css"/>
</head>
<body>
  <div class="container">
    <h1>🧬 Gene Expression Cancer Classifier</h1>
    <p class="subtitle">Upload a gene expression profile to predict cancer type</p>

    <div class="card">
      <h2>Input Options</h2>
      <!-- Option A: Upload CSV -->
      <label>Upload Sample CSV (1 row × N genes):</label>
      <input type="file" id="csvFile" accept=".csv"/>

      <!-- Option B: Sample Data -->
      <button id="loadSample">Load Demo Sample</button>

      <div id="inputPreview" class="preview"></div>
      <button id="predictBtn" class="btn-primary" disabled>🔬 Predict Cancer Type</button>
    </div>

    <div class="card result-card" id="resultCard" style="display:none">
      <h2>Prediction Result</h2>
      <div class="prediction-badge" id="predBadge"></div>
      <div class="confidence" id="confidence"></div>
      <h3>Class Probabilities</h3>
      <div id="probChart"></div>
      <h3>Top Expressed Genes</h3>
      <div id="topGenes"></div>
    </div>
  </div>
  <script src="app.js"></script>
</body>
</html>
```

---

## 🚀 TRAINING SCRIPT — `scripts/train.py`

```python
# scripts/train.py
"""
CLI training script.
Usage: python scripts/train.py
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data.preprocess import preprocess
from src.models.deep_net import CancerClassifierDNN
from src.training.trainer import Trainer
from src.training.evaluate import full_evaluation
from src.utils.config import CANCER_CLASSES, RANDOM_STATE

def main():
    print("🔄 Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, le, features = preprocess(save=True)

    print(f"✅ Data ready. Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Classes: {le.classes_}")

    # Validation split from train
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
    )

    print("\n🧠 Building model...")
    model = CancerClassifierDNN(input_dim=X_train.shape[1])

    print("\n🏋️ Training...")
    trainer = Trainer(model)
    history = trainer.train(X_tr, y_tr, X_val, y_val)

    print("\n📊 Final Evaluation on Test Set:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best checkpoint
    from src.utils.config import MODELS_DIR
    model.load_state_dict(torch.load(MODELS_DIR / "deep_net_best.pth", map_location=device))

    full_evaluation(model, X_test, y_test, list(le.classes_), device)

if __name__ == "__main__":
    main()
```

---

## 🐳 DOCKERFILE

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🐳 DOCKER COMPOSE

```yaml
# docker-compose.yml
version: "3.9"

services:
  classifier:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
```

---

## 📈 EXPECTED RESULTS (BENCHMARKS)

| Model | Accuracy | F1 (macro) | Notes |
|---|---|---|---|
| Logistic Regression | ~88% | ~87% | Baseline |
| Random Forest | ~94% | ~93% | Good baseline |
| XGBoost | ~95% | ~94% | Strong baseline |
| MLP (2 layers) | ~96% | ~95% | Simple NN |
| **Deep DNN (4 layers)** | **~98%** | **~97%** | **Main model** |

---

## 🗓️ IMPLEMENTATION TIMELINE

| Day | Task |
|---|---|
| Day 1 | Set up project structure, download dataset, EDA notebook |
| Day 2 | Build and test preprocessing pipeline (`preprocess.py`) |
| Day 3 | Train baseline models (RF, XGBoost), benchmark results |
| Day 4 | Build DNN model, training loop, early stopping |
| Day 5 | Evaluation (confusion matrix, ROC curves, PCA plots) |
| Day 6 | FastAPI backend + inference engine |
| Day 7 | Frontend UI + integration testing |
| Day 8 | Dockerize + final testing + README + slides |

---

## ✅ CODEX AGENT INSTRUCTIONS

When implementing this blueprint, follow this exact order:

1. **Create** the full directory structure as specified
2. **Install** all dependencies from `requirements.txt`
3. **Implement** `src/utils/config.py` first — all other files import from it
4. **Implement** `src/data/preprocess.py` — run and verify output shapes
5. **Implement** `src/models/deep_net.py` — verify forward pass with dummy tensor
6. **Implement** `src/training/trainer.py` — run one epoch smoke test
7. **Implement** `src/training/evaluate.py` — verify plot generation
8. **Run** `scripts/train.py` — full training run
9. **Implement** `app/inference.py` — verify prediction output shape
10. **Implement** `app/main.py` — start server, test `/predict` endpoint
11. **Implement** `app/frontend/` — connect UI to API
12. **Write** tests in `tests/` — pytest all modules
13. **Dockerize** — build image, run container, verify end-to-end

**Key constraints:**
- NEVER hardcode paths — always use `config.py` constants
- ALL model artifacts (scaler, encoder, feature names) MUST be saved during training
- The inference engine MUST load artifacts from saved paths, not from memory
- API must validate input length against `feature_names` list
- Use `torch.no_grad()` in all inference code

# Missing Artifacts — Regeneration Guide

The following pickle-serialized binary files were removed from this archive to avoid Gmail blocking:

| Removed File | Regenerate By |
|---|---|
| `models/artifacts/scaler.pkl` | `python src/data/preprocess.py` |
| `models/artifacts/label_encoder.pkl` | `python src/data/preprocess.py` |
| `models/artifacts/pca_reducer.pkl` | `python src/data/preprocess.py` |
| `models/saved/deep_net_best.pth` | `python scripts/train.py` |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data
python -m src.data.download

# 3. Preprocess data and generate artifacts (scaler.pkl, label_encoder.pkl, pca_reducer.pkl)
python src/data/preprocess.py

# 4. Train the model (generates deep_net_best.pth)
python scripts/train.py

# 5. Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

All artifacts will be regenerated in `models/artifacts/` and `models/saved/` with identical structure.

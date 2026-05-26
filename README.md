# Gene Expression Cancer Classifier

Full-stack ML project for classifying cancer type from RNA-Seq gene expression profiles. It includes a UCI dataset downloader, preprocessing pipeline, PyTorch DNN trainer, FastAPI inference API, vanilla frontend, tests, and Docker setup.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

## Data

Download the UCI RNA-Seq archive into `data/raw/`:

```bash
python -m src.data.download
```

Expected raw files:

- `data/raw/TCGA-PANCAN-HiSeq-801x20531/data.csv`
- `data/raw/TCGA-PANCAN-HiSeq-801x20531/labels.csv`

## Train

```bash
python scripts/train.py
```

Training saves:

- `models/saved/deep_net_best.pth`
- `models/artifacts/scaler.pkl`
- `models/artifacts/label_encoder.pkl`
- `models/artifacts/feature_names.json`
- `models/artifacts/training_manifest.json`
- processed NumPy arrays in `data/processed/`

Text evaluation runs by default. To also write Matplotlib report plots, run:

```bash
python scripts/train.py --plots
python scripts/evaluate.py --plots
```

Run stratified cross-validation benchmarks:

```bash
python scripts/cross_validate.py
```

Run DNN hyperparameter search:

```bash
python scripts/tune_dnn.py --trials 12 --epochs 25
```

## Run The API

```bash
uvicorn app.main:app --reload
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- Frontend: `http://127.0.0.1:8000/frontend/index.html`

If model artifacts are missing, the API still starts and `/health` reports what must be created before prediction.
The API also exposes `/metadata` with model version and confidence-threshold settings.

## Predict From CLI

```bash
python scripts/predict.py --csv path/to/sample.csv
```

The CSV should contain one sample row with the selected feature count produced by preprocessing.

## Test

```bash
pytest
```

## Docker

```bash
docker compose up --build
```

The `data/` and `models/` folders are mounted so downloaded data and trained artifacts persist outside the container.

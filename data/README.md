# Data

Run the downloader from the repository root:

```bash
python -m src.data.download
```

The default source is the UCI Gene Expression Cancer RNA-Seq archive. The preprocessing pipeline expects:

- `raw/data.csv`
- `raw/labels.csv`

Processed arrays are written to `processed/` by `python scripts/train.py`.

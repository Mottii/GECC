import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from app.inference import CancerInferenceEngine


def _values_from_csv(path: Path) -> list[float]:
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError(f"No numeric columns found in {path}")
    return numeric.iloc[0].astype(float).tolist()


def _values_from_json(raw: str) -> list[float]:
    payload = json.loads(raw)
    if isinstance(payload, dict):
        payload = payload.get("gene_values")
    if not isinstance(payload, list):
        raise ValueError("JSON input must be a list or an object with a gene_values list.")
    return [float(value) for value in payload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict cancer type for one sample.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV containing one numeric sample row.")
    source.add_argument("--json", help="JSON list, or object with gene_values.")
    parser.add_argument("--sample-id", default="cli_sample")
    args = parser.parse_args()

    values = _values_from_csv(args.csv) if args.csv else _values_from_json(args.json)
    engine = CancerInferenceEngine()
    if len(values) != len(engine.feature_names):
        raise ValueError(f"Expected {len(engine.feature_names)} features, got {len(values)}")

    result = engine.predict(values)
    print(json.dumps({"sample_id": args.sample_id, **result}, indent=2))


if __name__ == "__main__":
    main()

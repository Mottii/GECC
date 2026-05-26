import json

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import preprocess


def test_preprocess_saves_arrays_and_artifacts(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"
    raw_dir.mkdir()

    rows = []
    labels = []
    classes = ["BRCA", "KIRC", "LUAD"]
    for idx in range(18):
        cls_index = idx % len(classes)
        rows.append(
            {
                "sample_id": f"S{idx:03d}",
                "gene_a": idx + cls_index,
                "gene_b": idx * 2,
                "gene_c": 1,
                "gene_d": cls_index * 4 + idx / 10,
                "gene_e": idx % 5,
            }
        )
        labels.append({"sample_id": f"S{idx:03d}", "Class": classes[cls_index]})

    pd.DataFrame(rows).to_csv(raw_dir / "data.csv", index=False)
    pd.DataFrame(labels).to_csv(raw_dir / "labels.csv", index=False)

    X_train, X_test, y_train, y_test, scaler, encoder, feature_names = preprocess(
        save=True,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        artifacts_dir=artifacts_dir,
        top_k_features=3,
        test_size=0.33,
    )

    assert X_train.shape[1] == 3
    assert X_test.shape[1] == 3
    assert len(y_train) + len(y_test) == 18
    assert scaler is not None
    assert set(encoder.classes_) == set(classes)
    assert len(feature_names) == 3
    assert (processed_dir / "X_train.npy").exists()
    assert (processed_dir / "X_test.npy").exists()
    assert (artifacts_dir / "scaler.pkl").exists()
    assert (artifacts_dir / "label_encoder.pkl").exists()
    assert np.load(processed_dir / "y_test.npy").shape == y_test.shape
    assert joblib.load(artifacts_dir / "label_encoder.pkl").classes_.tolist() == sorted(classes)
    assert json.loads((artifacts_dir / "feature_names.json").read_text()) == feature_names

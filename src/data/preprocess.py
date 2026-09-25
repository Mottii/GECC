import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.config import (
    ARTIFACTS_DIR,
    NORMALIZE,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DIR,
    TEST_SIZE,
    TOP_K_FEATURES,
)


def resolve_raw_paths(raw_dir: Path = RAW_DIR) -> tuple[Path, Path]:
    data_path = raw_dir / "data.csv"
    labels_path = raw_dir / "labels.csv"
    if data_path.exists() and labels_path.exists():
        return data_path, labels_path

    data_candidates = sorted(raw_dir.rglob("data.csv"))
    labels_candidates = sorted(raw_dir.rglob("labels.csv"))
    if data_candidates and labels_candidates:
        return data_candidates[0], labels_candidates[0]

    raise FileNotFoundError(
        f"Expected raw files named data.csv and labels.csv somewhere under {raw_dir}. "
        "Run `python -m src.data.download` first."
    )


def _read_csv_with_optional_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    first_col = df.columns[0]
    if first_col.lower().startswith("unnamed") or first_col.lower() in {"sample", "sample_id", "id"}:
        df = df.set_index(first_col)
    return df


def load_raw_data(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.Series]:
    data_path, labels_path = resolve_raw_paths(raw_dir)
    X = _read_csv_with_optional_index(data_path)
    labels_df = _read_csv_with_optional_index(labels_path)

    label_col = "Class" if "Class" in labels_df.columns else labels_df.columns[-1]
    y = labels_df[label_col].astype(str)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    if len(X) != len(y):
        common_index = X.index.intersection(y.index)
        if common_index.empty:
            raise ValueError(f"Data rows ({len(X)}) and labels ({len(y)}) do not align.")
        X = X.loc[common_index]
        y = y.loc[common_index]

    return X, y


def select_features(X: pd.DataFrame, k: int = TOP_K_FEATURES) -> pd.DataFrame:
    k = min(k, X.shape[1])
    variances = X.var(axis=0)
    top_genes = variances.nlargest(k).index
    return X.loc[:, top_genes]


def prepare_full_dataset(
    raw_dir: Path = RAW_DIR,
    top_k_features: int = TOP_K_FEATURES,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, list[str]]:
    X, y = load_raw_data(raw_dir=raw_dir)
    selector = VarianceThreshold()
    X_filtered = pd.DataFrame(
        selector.fit_transform(X),
        columns=X.columns[selector.get_support()],
        index=X.index,
    )
    X_selected = select_features(X_filtered, k=top_k_features)
    feature_names = X_selected.columns.tolist()

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return X_selected.to_numpy(dtype=np.float32), y_encoded, encoder, feature_names


def preprocess(
    save: bool = True,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    top_k_features: int = TOP_K_FEATURES,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler | None, LabelEncoder, list[str]]:
    # Step 1: Load raw expression and labels
    X, y = load_raw_data(raw_dir=raw_dir)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Step 2: Split BEFORE feature selection to eliminate data leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    # Step 3: Fit VarianceThreshold strictly on training split
    selector = VarianceThreshold()
    selector.fit(X_train_raw)
    X_train_filtered = X_train_raw.loc[:, selector.get_support()]

    # Step 4: Select top-k variance genes strictly from training split
    k = min(top_k_features, X_train_filtered.shape[1])
    variances = X_train_filtered.var(axis=0)
    top_genes = variances.nlargest(k).index.tolist()
    feature_names = top_genes

    # Step 5: Transform train and test using the fitted feature set
    X_train = X_train_filtered[top_genes].to_numpy(dtype=np.float32)
    X_test = X_test_raw[top_genes].to_numpy(dtype=np.float32)

    scaler: StandardScaler | None = None
    if NORMALIZE:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    # Fit PCA for 2D visualization on training data
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train).astype(np.float32)
    X_test_pca = pca.transform(X_test).astype(np.float32)

    if save:
        processed_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        np.save(processed_dir / "X_train.npy", X_train)
        np.save(processed_dir / "X_test.npy", X_test)
        np.save(processed_dir / "y_train.npy", y_train)
        np.save(processed_dir / "y_test.npy", y_test)
        if scaler is not None:
            joblib.dump(scaler, artifacts_dir / "scaler.pkl")
        joblib.dump(encoder, artifacts_dir / "label_encoder.pkl")
        
        # Save PCA reducer & reference coordinates dynamically relative to artifacts_dir
        joblib.dump(pca, artifacts_dir / "pca_reducer.pkl")
        
        ref_points = [
            {"x": float(coord[0]), "y": float(coord[1]), "label": str(encoder.classes_[y_val_item])}
            for coord, y_val_item in zip(X_test_pca, y_test)
        ]
        with (artifacts_dir / "pca_reference_points.json").open("w", encoding="utf-8") as handle:
            json.dump(ref_points, handle, indent=2)

        with (artifacts_dir / "feature_names.json").open("w", encoding="utf-8") as handle:
            json.dump(feature_names, handle, indent=2)
        print(f"Saved processed data. Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler, encoder, feature_names

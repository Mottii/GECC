import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data.preprocess import load_raw_data, prepare_full_dataset
from src.models.baseline import make_baselines
from src.models.deep_net import CancerClassifierDNN
from src.training.trainer import Trainer
from src.utils.config import CV_RESULTS_PATH, RANDOM_STATE, TOP_K_FEATURES
from src.utils.manifest import write_json_manifest
from src.utils.reproducibility import set_global_seed


def aggregate(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def select_fold_features(
    X_tr: pd.DataFrame | np.ndarray,
    X_te: pd.DataFrame | np.ndarray,
    top_k: int = TOP_K_FEATURES,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(X_tr, pd.DataFrame):
        selector = VarianceThreshold()
        selector.fit(X_tr)
        train_filtered = X_tr.loc[:, selector.get_support()]
        k = min(top_k, train_filtered.shape[1])
        top_genes = train_filtered.var(axis=0).nlargest(k).index.tolist()
        return train_filtered[top_genes].to_numpy(dtype=np.float32), X_te[top_genes].to_numpy(dtype=np.float32)
    return np.asarray(X_tr, dtype=np.float32), np.asarray(X_te, dtype=np.float32)


def run_baseline_cv(X: pd.DataFrame | np.ndarray, y: np.ndarray, folds: int, seed: int) -> dict[str, dict]:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results: dict[str, dict] = {}
    for name, model in make_baselines(random_state=seed).items():
        accs: list[float] = []
        f1s: list[float] = []
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = select_fold_features(
                X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx],
                X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx],
            )
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", clone(model))])
            pipeline.fit(X_tr, y[train_idx])
            preds = pipeline.predict(X_te)
            accs.append(float(accuracy_score(y[test_idx], preds)))
            f1s.append(float(f1_score(y[test_idx], preds, average="macro")))
        results[name] = {
            "accuracy": aggregate(accs),
            "f1_macro": aggregate(f1s),
            "fold_accuracies": accs,
            "fold_f1_macro": f1s,
        }
    return results


def run_dnn_cv(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    class_count: int,
    folds: int,
    seed: int,
    epochs: int,
    batch_size: int,
) -> dict:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs: list[float] = []
    f1s: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr_fold, X_test_fold = select_fold_features(
            X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx],
            X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx],
        )
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_fold,
            y_train_fold,
            test_size=0.15,
            random_state=seed + fold_idx,
            stratify=y_train_fold,
        )

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test_fold = scaler.transform(X_test_fold).astype(np.float32)

        model = CancerClassifierDNN(input_dim=X_tr.shape[1], num_classes=class_count)
        fold_model_dir = Path(tempfile.mkdtemp(prefix=f"dnn_cv_fold_{fold_idx}_"))
        trainer = Trainer(model, model_dir=fold_model_dir, seed=seed + fold_idx)
        trainer.train(X_tr, y_tr, X_val, y_val, epochs=epochs, batch_size=batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(fold_model_dir / "deep_net_best.pth", map_location=device))
        model.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test_fold, dtype=torch.float32).to(device))
            preds = logits.argmax(dim=1).cpu().numpy()

        accs.append(float(accuracy_score(y_test_fold, preds)))
        f1s.append(float(f1_score(y_test_fold, preds, average="macro")))
        print(f"[DNN CV] Fold {fold_idx}/{folds}: acc={accs[-1]:.4f}, f1_macro={f1s[-1]:.4f}")

    return {
        "accuracy": aggregate(accs),
        "f1_macro": aggregate(f1s),
        "fold_accuracies": accs,
        "fold_f1_macro": f1s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stratified K-fold cross-validation.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--dnn-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-dnn", action="store_true", help="Only evaluate sklearn baselines.")
    parser.add_argument("--preselected", action="store_true", help="Use legacy preselected features instead of raw.")
    args = parser.parse_args()

    if args.folds < 3:
        raise ValueError("folds must be >= 3")

    set_global_seed(args.seed)
    if not args.preselected:
        X, y_series = load_raw_data()
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_series)
        feature_count = TOP_K_FEATURES
        print(f"Loaded raw dataset for leak-free CV: X={X.shape}, classes={list(encoder.classes_)}")
    else:
        X, y, encoder, feature_names = prepare_full_dataset()
        feature_count = len(feature_names)
        print(f"Loaded preselected dataset: X={X.shape}, classes={list(encoder.classes_)}")

    baseline_results = run_baseline_cv(X, y, folds=args.folds, seed=args.seed)
    dnn_results = None
    if not args.skip_dnn:
        dnn_results = run_dnn_cv(
            X,
            y,
            class_count=len(encoder.classes_),
            folds=args.folds,
            seed=args.seed,
            epochs=args.dnn_epochs,
            batch_size=args.batch_size,
        )

    payload = {
        "seed": args.seed,
        "folds": args.folds,
        "feature_count": feature_count,
        "classes": [str(c) for c in encoder.classes_],
        "baseline_results": baseline_results,
        "dnn_results": dnn_results,
    }
    manifest = write_json_manifest(CV_RESULTS_PATH, payload)
    print(f"Saved cross-validation report to {CV_RESULTS_PATH}")
    print(json.dumps({"run_id": manifest["run_id"], "folds": args.folds}, indent=2))


if __name__ == "__main__":
    main()

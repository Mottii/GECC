from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency/runtime can be missing in CI.
    XGBClassifier = None


@dataclass
class BaselineResult:
    name: str
    accuracy: float
    f1_macro: float


def make_baselines(random_state: int = 42) -> dict[str, object]:
    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "svm": SVC(kernel="rbf", probability=True, random_state=random_state),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
        )
    return models


def train_and_evaluate_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir=None,
) -> list[BaselineResult]:
    results: list[BaselineResult] = []
    for name, model in make_baselines().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results.append(
            BaselineResult(
                name=name,
                accuracy=float(accuracy_score(y_test, preds)),
                f1_macro=float(f1_score(y_test, preds, average="macro")),
            )
        )
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, save_dir / f"{name}.pkl")
    return results

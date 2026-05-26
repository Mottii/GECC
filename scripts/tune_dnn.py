import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from src.data.preprocess import prepare_full_dataset
from src.models.deep_net import CancerClassifierDNN
from src.utils.config import RANDOM_STATE, TUNING_RESULTS_PATH
from src.utils.manifest import write_json_manifest
from src.utils.reproducibility import set_global_seed


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_trial(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    params: dict,
    seed: int,
) -> tuple[float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CancerClassifierDNN(
        input_dim=X_train.shape[1],
        hidden_layers=params["hidden_layers"],
        num_classes=num_classes,
        dropout_rate=params["dropout_rate"],
    )
    model.to(device)

    train_loader = make_loader(X_train, y_train, params["batch_size"], seed, shuffle=True)
    val_loader = make_loader(X_val, y_val, params["batch_size"], seed + 1, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, params["epochs"]))

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_state = None
    for _ in range(params["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                targets.extend(yb.numpy().tolist())
        val_acc = float(accuracy_score(targets, preds))
        val_f1 = float(f1_score(targets, preds, average="macro"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc, best_val_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-search hyperparameter tuning for the DNN classifier.")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--epochs", type=int, default=25, help="Epochs per trial.")
    args = parser.parse_args()

    set_global_seed(args.seed)

    X, y, encoder, feature_names = prepare_full_dataset()
    X_train_full, X_test_raw, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=args.seed + 1,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    search_space = {
        "hidden_layers": [[512, 256], [1024, 512, 256], [512, 256, 128], [1024, 512, 256, 128]],
        "dropout_rate": [0.2, 0.3, 0.4, 0.5],
        "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "batch_size": [16, 32, 64],
    }
    rng = random.Random(args.seed)
    sampled = list(ParameterSampler(search_space, n_iter=args.trials, random_state=rng.randint(0, 10_000)))

    trials = []
    best_trial = None
    for idx, params in enumerate(sampled, start=1):
        params = dict(params)
        params["epochs"] = args.epochs
        val_acc, val_f1 = train_trial(
            X_train,
            y_train,
            X_val,
            y_val,
            num_classes=len(encoder.classes_),
            params=params,
            seed=args.seed + idx,
        )
        trial = {"trial": idx, "params": params, "val_accuracy": val_acc, "val_f1_macro": val_f1}
        trials.append(trial)
        if best_trial is None or trial["val_accuracy"] > best_trial["val_accuracy"]:
            best_trial = trial
        print(f"Trial {idx}/{args.trials}: val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

    if best_trial is None:
        raise RuntimeError("No tuning trials were executed.")

    # Evaluate best parameterization once on held-out test split.
    best_params = dict(best_trial["params"])
    scaler_full = StandardScaler()
    X_train_full_scaled = scaler_full.fit_transform(X_train_full).astype(np.float32)
    X_test_scaled = scaler_full.transform(X_test_raw).astype(np.float32)

    test_acc, test_f1 = train_trial(
        X_train_full_scaled,
        y_train_full,
        X_test_scaled,
        y_test,
        num_classes=len(encoder.classes_),
        params=best_params,
        seed=args.seed + 999,
    )

    payload = {
        "seed": args.seed,
        "trials": args.trials,
        "feature_count": len(feature_names),
        "classes": [str(c) for c in encoder.classes_],
        "best_trial": best_trial,
        "test_estimate_from_best_params": {"accuracy": test_acc, "f1_macro": test_f1},
        "all_trials": trials,
    }
    manifest = write_json_manifest(TUNING_RESULTS_PATH, payload)
    print(f"Saved tuning report to {TUNING_RESULTS_PATH}")
    print(json.dumps({"run_id": manifest["run_id"], "best_val_accuracy": best_trial["val_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from sklearn.model_selection import train_test_split

from src.data.preprocess import preprocess
from src.models.deep_net import CancerClassifierDNN
from src.training.trainer import Trainer
from src.utils.config import MANIFEST_PATH, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE
from src.utils.manifest import environment_snapshot, hash_feature_names, write_json_manifest
from src.utils.reproducibility import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the gene expression DNN classifier.")
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override configured batch size.")
    parser.add_argument("--plots", action="store_true", help="Write evaluation plots after training.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Global random seed.")
    args = parser.parse_args()
    set_global_seed(args.seed)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, _, encoder, features = preprocess(save=True, random_state=args.seed)
    print(f"Data ready. Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Classes: {list(encoder.classes_)}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=args.seed,
        stratify=y_train,
    )

    print("Building model...")
    model = CancerClassifierDNN(input_dim=X_train.shape[1], num_classes=len(encoder.classes_))

    print("Training...")
    trainer = Trainer(model, seed=args.seed)
    train_kwargs = {}
    if args.epochs is not None:
        train_kwargs["epochs"] = args.epochs
    if args.batch_size is not None:
        train_kwargs["batch_size"] = args.batch_size
    history = trainer.train(X_tr, y_tr, X_val, y_val, **train_kwargs)

    print("Final evaluation on test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODELS_DIR / "deep_net_best.pth", map_location=device))
    model.to(device)
    output_dir = PROCESSED_DIR / "reports" if args.plots else None
    from src.training.evaluate import full_evaluation

    y_pred, _, _ = full_evaluation(model, X_test, y_test, list(encoder.classes_), device, output_dir=output_dir)

    manifest = write_json_manifest(
        MANIFEST_PATH,
        {
            "seed": args.seed,
            "dataset": {
                "train_shape": list(X_train.shape),
                "test_shape": list(X_test.shape),
                "feature_count": len(features),
                "feature_hash_sha256": hash_feature_names(features),
                "classes": [str(c) for c in encoder.classes_],
            },
            "training": {
                "epochs_configured": args.epochs,
                "batch_size_configured": args.batch_size,
                "epochs_ran": history["epochs_ran"][0],
                "best_val_acc": history["best_val_acc"][0],
            },
            "evaluation": {
                "test_accuracy": float((y_pred == y_test).mean()),
                "test_samples": int(y_test.shape[0]),
            },
            "artifacts": {
                "model_path": str(MODELS_DIR / "deep_net_best.pth"),
                "manifest_path": str(MANIFEST_PATH),
            },
            "environment": environment_snapshot(),
        },
    )
    print(f"Saved training manifest: {MANIFEST_PATH}")
    print(json.dumps({"run_id": manifest["run_id"], "test_accuracy": manifest["evaluation"]["test_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()

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

    # Post-hoc Temperature Scaling Calibration
    print("Calibrating softmax temperature...")
    calibrated_temperature = model.calibrate_temperature(
        torch.tensor(X_val, dtype=torch.float32),
        y_val,
        device=device,
    )
    from src.utils.config import DEFAULT_TEMPERATURE
    if calibrated_temperature < DEFAULT_TEMPERATURE:
        calibrated_temperature = DEFAULT_TEMPERATURE
        model.set_temperature(calibrated_temperature)

    from src.utils.config import ARTIFACTS_DIR
    with (ARTIFACTS_DIR / "temperature.json").open("w", encoding="utf-8") as handle:
        json.dump({"temperature": calibrated_temperature}, handle, indent=2)
    print(f"Saved calibrated temperature ({calibrated_temperature}) to {ARTIFACTS_DIR / 'temperature.json'}")

    # Train Autoencoder for OOD Gating
    print("Training Autoencoder for Out-Of-Distribution (OOD) gating...")
    from src.models.autoencoder import train_autoencoder
    ae_model, ood_threshold = train_autoencoder(X_train, input_dim=X_train.shape[1], epochs=25, device=device)
    torch.save(ae_model.state_dict(), MODELS_DIR / "autoencoder.pth")
    with (ARTIFACTS_DIR / "ood_threshold.json").open("w", encoding="utf-8") as handle:
        json.dump({"ood_threshold": round(ood_threshold, 4)}, handle, indent=2)
    print(f"Autoencoder saved (OOD threshold: {ood_threshold:.4f})")

    # Extract Curated Real Patient Demo Profiles
    print("Extracting curated real patient demo profiles...")
    from src.data.preprocess import load_raw_data
    raw_df, raw_labels = load_raw_data()
    
    # Sample A: Classical BRCA
    brca_id = raw_labels[raw_labels == "BRCA"].index[0]
    sample_a_vals = [float(x) for x in raw_df.loc[brca_id, features]]
    
    # Sample B: Classical KIRC
    kirc_id = raw_labels[raw_labels == "KIRC"].index[0]
    sample_b_vals = [float(x) for x in raw_df.loc[kirc_id, features]]
    
    # Sample C: Ambiguous Sample 46 (LUAD vs BRCA borderline)
    sample_46_id = "sample_129" if "sample_129" in raw_df.index else raw_labels[raw_labels == "LUAD"].index[0]
    sample_c_vals = [float(x) for x in raw_df.loc[sample_46_id, features]]

    demo_profiles = {
        "brca": {
            "name": "Sample A: Classical BRCA (Breast Carcinoma)",
            "cohort": "BRCA",
            "description": "Classical invasive breast carcinoma profile with robust estrogen/progesterone pathway and cytokeratin expression.",
            "gene_values": sample_a_vals,
        },
        "kirc": {
            "name": "Sample B: Classical KIRC (Kidney Renal Clear Cell)",
            "cohort": "KIRC",
            "description": "Classical clear cell renal carcinoma showing characteristic VHL inactivation and angiogenic signatures.",
            "gene_values": sample_b_vals,
        },
        "sample46": {
            "name": "Sample C: Ambiguous Sample 46 (LUAD vs BRCA)",
            "cohort": "LUAD",
            "description": "Borderline lung adenocarcinoma biopsy with overlapping glandular features. Softened via temperature scaling to flag clinician uncertainty.",
            "gene_values": sample_c_vals,
        },
    }

    with (ARTIFACTS_DIR / "demo_profiles.json").open("w", encoding="utf-8") as handle:
        json.dump(demo_profiles, handle, indent=2)
    
    # Also save to frontend directory for immediate web serving
    frontend_demo = ROOT / "app" / "frontend" / "demo_profiles.json"
    with frontend_demo.open("w", encoding="utf-8") as handle:
        json.dump(demo_profiles, handle, indent=2)
    print(f"Saved curated demo profiles to {frontend_demo}")

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
                "best_val_loss": history["best_val_loss"][0],
                "calibrated_temperature": calibrated_temperature,
                "ood_threshold": ood_threshold,
            },
            "evaluation": {
                "test_accuracy": float((y_pred == y_test).mean()),
                "test_samples": int(y_test.shape[0]),
            },
            "artifacts": {
                "model_path": str(MODELS_DIR / "deep_net_best.pth"),
                "autoencoder_path": str(MODELS_DIR / "autoencoder.pth"),
                "manifest_path": str(MANIFEST_PATH),
            },
            "environment": environment_snapshot(),
        },
    )
    print(f"Saved training manifest: {MANIFEST_PATH}")
    print(json.dumps({"run_id": manifest["run_id"], "test_accuracy": manifest["evaluation"]["test_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()

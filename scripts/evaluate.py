import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report

from src.models.deep_net import CancerClassifierDNN
from src.utils.config import ENCODER_PATH, FEATURES_PATH, MODEL_PATH, PROCESSED_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved DNN classifier.")
    parser.add_argument("--plots", action="store_true", help="Write confusion matrix, ROC, and PCA plots.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    args = parser.parse_args()

    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    encoder = joblib.load(ENCODER_PATH)

    import json

    with FEATURES_PATH.open(encoding="utf-8") as handle:
        feature_names = json.load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CancerClassifierDNN(input_dim=len(feature_names), num_classes=len(encoder.classes_))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    if args.plots or args.show:
        from src.training.evaluate import full_evaluation

        full_evaluation(
            model,
            X_test,
            y_test,
            list(encoder.classes_),
            device,
            output_dir=PROCESSED_DIR / "reports" if args.plots else None,
            show=args.show,
        )
        return

    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()
    print(classification_report(y_test, y_pred, target_names=list(encoder.classes_)))


if __name__ == "__main__":
    main()

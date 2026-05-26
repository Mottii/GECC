import json
from pathlib import Path

import joblib
import numpy as np
import torch

from src.models.deep_net import CancerClassifierDNN
from src.utils.config import (
    CANCER_CLASSES,
    CONFIDENCE_THRESHOLD,
    DROPOUT_RATE,
    ENCODER_PATH,
    FEATURES_PATH,
    HIDDEN_LAYERS,
    MANIFEST_PATH,
    MAX_INPUT_ABS_VALUE,
    MODEL_PATH,
    SCALER_PATH,
)


class ArtifactNotReadyError(RuntimeError):
    """Raised when prediction is attempted before training artifacts exist."""


class InvalidInputError(ValueError):
    """Raised when incoming feature values are malformed or unsafe."""


class CancerInferenceEngine:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        encoder_path: Path = ENCODER_PATH,
        features_path: Path = FEATURES_PATH,
        manifest_path: Path = MANIFEST_PATH,
        device: torch.device | None = None,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.encoder_path = Path(encoder_path)
        self.features_path = Path(features_path)
        self.manifest_path = Path(manifest_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.load_error: str | None = None
        self.class_names: list[str] = CANCER_CLASSES.copy()
        self.feature_names: list[str] = []
        self.model: CancerClassifierDNN | None = None
        self.scaler = None
        self.encoder = None
        self.manifest: dict | None = None
        self._load_artifacts()

    def _missing_artifacts(self) -> list[Path]:
        return [
            path
            for path in (self.model_path, self.scaler_path, self.encoder_path, self.features_path)
            if not path.exists()
        ]

    def _load_artifacts(self) -> None:
        missing = self._missing_artifacts()
        if missing:
            self.load_error = "Missing artifacts: " + ", ".join(str(path) for path in missing)
            return

        try:
            self.scaler = joblib.load(self.scaler_path)
            self.encoder = joblib.load(self.encoder_path)
            self.class_names = [str(cls) for cls in self.encoder.classes_]
            with self.features_path.open(encoding="utf-8") as handle:
                self.feature_names = json.load(handle)

            self.model = CancerClassifierDNN(
                input_dim=len(self.feature_names),
                hidden_layers=HIDDEN_LAYERS,
                num_classes=len(self.class_names),
                dropout_rate=DROPOUT_RATE,
            )
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            if self.manifest_path.exists():
                with self.manifest_path.open(encoding="utf-8") as handle:
                    self.manifest = json.load(handle)
            else:
                self.manifest = None
            self.model_loaded = True
            self.load_error = None
        except Exception as exc:  # pragma: no cover - exact loader errors vary by torch/joblib version.
            self.model_loaded = False
            self.load_error = f"Failed to load artifacts: {exc}"

    def reload(self) -> None:
        self.model_loaded = False
        self.load_error = None
        self.manifest = None
        self._load_artifacts()

    def readiness_message(self) -> str:
        if self.model_loaded:
            return "Model artifacts loaded."
        return (
            f"{self.load_error}. Run `python -m src.data.download` and "
            "`python scripts/train.py` to create prediction artifacts."
        )

    def model_metadata(self) -> dict:
        run_id = self.current_model_version()
        trained_at = self.manifest.get("created_at_utc") if self.manifest else None
        feature_hash = self.manifest.get("dataset", {}).get("feature_hash_sha256") if self.manifest else None
        test_acc = self.manifest.get("evaluation", {}).get("test_accuracy") if self.manifest else None
        return {
            "model_name": "CancerClassifierDNN",
            "model_loaded": self.model_loaded,
            "model_version": run_id,
            "trained_at_utc": trained_at,
            "feature_hash_sha256": feature_hash,
            "test_accuracy": test_acc,
            "feature_count": len(self.feature_names),
            "classes": self.class_names,
            "device": str(self.device),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "max_input_abs_value": MAX_INPUT_ABS_VALUE,
        }

    def current_model_version(self) -> str:
        if self.manifest and self.manifest.get("run_id"):
            return str(self.manifest["run_id"])
        if self.model_path.exists():
            timestamp = int(self.model_path.stat().st_mtime)
            return f"file-{timestamp}"
        return "unavailable"

    def _validate_input_values(self, gene_values: list[float]) -> np.ndarray:
        try:
            x = np.asarray(gene_values, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise InvalidInputError("gene_values must be a numeric list.") from exc

        if x.ndim != 1:
            raise InvalidInputError("gene_values must be a one-dimensional list.")
        if not np.isfinite(x).all():
            raise InvalidInputError("gene_values must not contain NaN or infinite values.")
        if np.abs(x).max(initial=0.0) > MAX_INPUT_ABS_VALUE:
            raise InvalidInputError(
                f"gene_values contains magnitude above {MAX_INPUT_ABS_VALUE:g}, which is rejected for safety."
            )
        return x

    def predict(self, gene_values: list[float]) -> dict:
        if not self.model_loaded or self.model is None or self.scaler is None:
            raise ArtifactNotReadyError(self.readiness_message())

        x = self._validate_input_values(gene_values).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            proba = self.model.predict_proba(x_tensor).cpu().numpy()[0]

        pred_idx = int(np.argmax(proba))
        pred_class = self.class_names[pred_idx]
        confidence = float(proba[pred_idx])
        low_confidence = confidence < CONFIDENCE_THRESHOLD
        warning = None
        if low_confidence:
            warning = (
                f"Confidence {confidence:.4f} is below threshold {CONFIDENCE_THRESHOLD:.2f}. "
                "Treat this prediction as uncertain."
            )
        class_probabilities = {
            cls: round(float(probability), 4)
            for cls, probability in zip(self.class_names, proba)
        }

        top_indices = np.argsort(x_scaled[0])[::-1][:5]
        top_features = [
            {"gene": self.feature_names[i], "expression": round(float(x_scaled[0][i]), 4)}
            for i in top_indices
        ]

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "class_probabilities": class_probabilities,
            "top_features": top_features,
            "low_confidence": low_confidence,
            "warning": warning,
            "model_version": self.current_model_version(),
        }

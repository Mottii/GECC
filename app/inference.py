import json
from pathlib import Path

import joblib
import numpy as np
import torch

from src.models.autoencoder import GeneExpressionAutoencoder
from src.models.deep_net import CancerClassifierDNN
from src.utils.config import (
    AUTOENCODER_PATH,
    CANCER_CLASSES,
    CONFIDENCE_THRESHOLD,
    DEFAULT_TEMPERATURE,
    DROPOUT_RATE,
    ENCODER_PATH,
    FEATURES_PATH,
    HIDDEN_LAYERS,
    MANIFEST_PATH,
    MAX_EXPRESSION_VALUE,
    MAX_INPUT_ABS_VALUE,
    MIN_EXPRESSION_VALUE,
    MODEL_PATH,
    OOD_THRESHOLD_PATH,
    PCA_REDUCER_PATH,
    PCA_REFERENCE_PATH,
    SCALER_PATH,
    TEMPERATURE_PATH,
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
        pca_reducer_path: Path = PCA_REDUCER_PATH,
        pca_reference_path: Path = PCA_REFERENCE_PATH,
        autoencoder_path: Path = AUTOENCODER_PATH,
        ood_threshold_path: Path = OOD_THRESHOLD_PATH,
        temperature_path: Path = TEMPERATURE_PATH,
        temperature: float = DEFAULT_TEMPERATURE,
        device: torch.device | None = None,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.encoder_path = Path(encoder_path)
        self.features_path = Path(features_path)
        self.manifest_path = Path(manifest_path)
        self.pca_reducer_path = Path(pca_reducer_path)
        self.pca_reference_path = Path(pca_reference_path)
        self.autoencoder_path = Path(autoencoder_path)
        self.ood_threshold_path = Path(ood_threshold_path)
        self.temperature_path = Path(temperature_path)
        if temperature is not None:
            self.temperature = float(temperature)
            self._explicit_temperature = True
        else:
            self.temperature = float(DEFAULT_TEMPERATURE)
            self._explicit_temperature = False
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.load_error: str | None = None
        self.class_names: list[str] = CANCER_CLASSES.copy()
        self.feature_names: list[str] = []
        self.model: CancerClassifierDNN | None = None
        self.autoencoder: GeneExpressionAutoencoder | None = None
        self.ood_threshold: float = 1.0
        self.scaler = None
        self.encoder = None
        self.manifest: dict | None = None
        self.pca_reducer = None
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

            # Load temperature calibration if present and not explicitly provided
            if not getattr(self, "_explicit_temperature", False) and self.temperature_path.exists():
                try:
                    with self.temperature_path.open(encoding="utf-8") as handle:
                        temp_data = json.load(handle)
                        self.temperature = float(temp_data.get("temperature", self.temperature))
                except Exception:
                    pass

            self.model = CancerClassifierDNN(
                input_dim=len(self.feature_names),
                hidden_layers=HIDDEN_LAYERS,
                num_classes=len(self.class_names),
                dropout_rate=DROPOUT_RATE,
                temperature=self.temperature,
            )
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.set_temperature(self.temperature)
            self.model.to(self.device)
            self.model.eval()

            # Load Autoencoder for OOD gating if present
            if self.autoencoder_path.exists():
                try:
                    self.autoencoder = GeneExpressionAutoencoder(input_dim=len(self.feature_names))
                    ae_state = torch.load(self.autoencoder_path, map_location=self.device)
                    self.autoencoder.load_state_dict(ae_state)
                    self.autoencoder.to(self.device)
                    self.autoencoder.eval()
                    if self.ood_threshold_path.exists():
                        with self.ood_threshold_path.open(encoding="utf-8") as handle:
                            self.ood_threshold = float(json.load(handle).get("ood_threshold", 1.0))
                except Exception:
                    self.autoencoder = None
            else:
                self.autoencoder = None

            # Load PCA reducer if exists
            if self.pca_reducer_path.exists():
                self.pca_reducer = joblib.load(self.pca_reducer_path)
            else:
                self.pca_reducer = None

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
            "min_expression_value": MIN_EXPRESSION_VALUE,
            "max_expression_value": MAX_EXPRESSION_VALUE,
            "max_input_abs_value": MAX_INPUT_ABS_VALUE,
            "temperature": self.temperature,
            "ood_gating_enabled": self.autoencoder is not None,
            "ood_threshold": self.ood_threshold if self.autoencoder else None,
        }

    def current_model_version(self) -> str:
        if self.manifest and self.manifest.get("run_id"):
            return str(self.manifest["run_id"])
        if self.model_path.exists():
            timestamp = int(self.model_path.stat().st_mtime)
            return f"file-{timestamp}"
        return "unavailable"

    def map_raw_features(self, raw_input: list[float] | dict[str, float] | np.ndarray) -> list[float]:
        """Align raw 20,531 features or dictionary to the model's selected feature space."""
        if isinstance(raw_input, dict):
            return [float(raw_input.get(g, 0.0)) for g in self.feature_names]
        if isinstance(raw_input, (list, np.ndarray)) and len(raw_input) == 20531:
            mapped = []
            for g in self.feature_names:
                if g.startswith("gene_") and g[5:].isdigit():
                    idx = int(g[5:])
                    if 0 <= idx < len(raw_input):
                        mapped.append(float(raw_input[idx]))
                    else:
                        mapped.append(0.0)
                else:
                    mapped.append(0.0)
            return mapped
        return list(raw_input)

    def _validate_input_values(self, gene_values: list[float]) -> np.ndarray:
        try:
            x = np.asarray(gene_values, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise InvalidInputError("gene_values must be a numeric list.") from exc

        if x.ndim != 1:
            raise InvalidInputError("gene_values must be a one-dimensional list.")
        if len(x) == 0:
            raise InvalidInputError("gene_values cannot be empty.")
        if len(self.feature_names) > 0 and len(x) != len(self.feature_names):
            raise InvalidInputError(
                f"Expected {len(self.feature_names)} features, but got {len(x)}."
            )
        if not np.isfinite(x).all():
            raise InvalidInputError("gene_values must not contain NaN or infinite values.")
        if (x < MIN_EXPRESSION_VALUE).any():
            raise InvalidInputError(
                f"gene_values contains negative values (< {MIN_EXPRESSION_VALUE}), which violate biological RNA-Seq bounds."
            )
        if (x > MAX_EXPRESSION_VALUE).any():
            raise InvalidInputError(
                f"gene_values contains values above biological maximum {MAX_EXPRESSION_VALUE:g}, which is rejected for safety."
            )
        return x

    def predict(self, gene_values: list[float] | dict[str, float]) -> dict:
        if not self.model_loaded or self.model is None or self.scaler is None:
            raise ArtifactNotReadyError(self.readiness_message())

        if isinstance(gene_values, dict) or (isinstance(gene_values, (list, np.ndarray)) and len(gene_values) == 20531):
            gene_values = self.map_raw_features(gene_values)

        x = self._validate_input_values(gene_values).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)

        # Calibrated Softmax Prediction using Temperature Scaling
        with torch.no_grad():
            proba = self.model.predict_proba(x_tensor, temperature=self.temperature).cpu().numpy()[0]

        pred_idx = int(np.argmax(proba))
        pred_class = self.class_names[pred_idx]
        confidence = float(proba[pred_idx])
        low_confidence = confidence < CONFIDENCE_THRESHOLD

        # Input x Gradient Feature Attribution (Positive Drivers and Negative Tumor Suppressors)
        self.model.eval()
        x_t_grad = torch.tensor(x_scaled, dtype=torch.float32, requires_grad=True, device=self.device)
        logits = self.model(x_t_grad)
        grad_tensor = torch.autograd.grad(
            outputs=logits[0, pred_idx],
            inputs=x_t_grad,
            retain_graph=False,
        )[0]
        grads = grad_tensor.detach().cpu().numpy()[0]
        attributions = x_scaled[0] * grads

        k_top = min(5, len(self.feature_names))
        pos_indices = np.argsort(attributions)[::-1][:k_top]
        top_features = [
            {
                "gene": self.feature_names[i],
                "expression": round(float(x_scaled[0][i]), 4),
                "attribution": round(float(attributions[i]), 4),
            }
            for i in pos_indices
        ]

        neg_indices = np.argsort(attributions)[:k_top]
        suppressed_features = [
            {
                "gene": self.feature_names[i],
                "expression": round(float(x_scaled[0][i]), 4),
                "attribution": round(float(attributions[i]), 4),
            }
            for i in neg_indices
            if attributions[i] < 0
        ]

        # Autoencoder OOD Gating
        is_all_zeros = np.allclose(x, 0.0)
        ood_score = None
        is_ood = False

        if self.autoencoder is not None:
            with torch.no_grad():
                rec_err = self.autoencoder.reconstruction_error(x_tensor).cpu().item()
                ood_score = round(float(rec_err), 4)
                if ood_score > self.ood_threshold:
                    is_ood = True
        elif is_all_zeros:
            is_ood = True
            ood_score = 999.0

        # Build clinical warnings
        warnings = []
        if low_confidence:
            warnings.append(
                f"Confidence {confidence:.4f} is below threshold {CONFIDENCE_THRESHOLD:.2f}. "
                "Treat this prediction as uncertain."
            )
        if is_ood:
            warnings.append(
                f"Sample flagged as Out-Of-Distribution (OOD score: {ood_score}). "
                "Expression profile does not resemble biological cancer cohorts."
            )
        warning = " | ".join(warnings) if warnings else None

        class_probabilities = {
            cls: round(float(probability), 4)
            for cls, probability in zip(self.class_names, proba)
        }

        # Calculate PCA coordinates
        pca_coords = {"x": 0.0, "y": 0.0}
        if self.pca_reducer is not None:
            coords = self.pca_reducer.transform(x_scaled)[0]
            pca_coords = {"x": float(coords[0]), "y": float(coords[1])}

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "class_probabilities": class_probabilities,
            "top_features": top_features,
            "suppressed_features": suppressed_features,
            "low_confidence": low_confidence,
            "warning": warning,
            "model_version": self.current_model_version(),
            "pca_coords": pca_coords,
            "ood_score": ood_score,
            "is_ood": is_ood,
        }

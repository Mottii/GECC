import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier

from app.inference import CancerInferenceEngine, InvalidInputError
from app.main import app, generate_local_fallback_response
from src.models.baseline import make_baselines
from src.models.deep_net import CancerClassifierDNN
from src.training.trainer import Trainer
from src.utils.config import (
    ARTIFACTS_DIR,
    AUTOENCODER_PATH,
    CONFIDENCE_THRESHOLD,
    ENCODER_PATH,
    FEATURES_PATH,
    MAX_EXPRESSION_VALUE,
    MIN_EXPRESSION_VALUE,
    MODEL_PATH,
    PCA_REDUCER_PATH,
    PCA_REFERENCE_PATH,
    SCALER_PATH,
    TEMPERATURE_PATH,
)


@pytest.fixture(scope="module")
def engine():
    eng = CancerInferenceEngine(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        encoder_path=ENCODER_PATH,
        features_path=FEATURES_PATH,
        pca_reducer_path=PCA_REDUCER_PATH,
        pca_reference_path=PCA_REFERENCE_PATH,
        autoencoder_path=AUTOENCODER_PATH,
        temperature_path=TEMPERATURE_PATH,
        device=torch.device("cpu"),
    )
    assert eng.model_loaded, f"Inference engine failed to load: {eng.load_error}"
    return eng


def test_zero_vector_flagged_as_ood(engine):
    """Edge Case 1: All-zero expression vector must be detected as Out-Of-Distribution."""
    zero_vector = [0.0] * len(engine.feature_names)
    result = engine.predict(zero_vector)

    assert result["is_ood"] is True, "Zero vector should be flagged as OOD."
    assert result["warning"] is not None
    assert "Out-Of-Distribution" in result["warning"]


def test_uniform_noise_rejected_or_flagged(engine):
    """Edge Case 2: Uniform random noise must be flagged as OOD or low-confidence."""
    rng = np.random.default_rng(42)
    noise_vector = rng.uniform(0.0, 15.0, size=len(engine.feature_names)).tolist()
    result = engine.predict(noise_vector)

    assert result["is_ood"] is True or result["low_confidence"] is True, (
        "Uniform noise must trigger OOD gating or low-confidence warning."
    )
    assert result["warning"] is not None


def test_inverted_expression_vector_rejected(engine):
    """Edge Case 3: Inverted expression vector must be rejected or flagged as OOD."""
    # Negative inversion must be strictly rejected
    inverted_negative = [-5.0] * len(engine.feature_names)
    with pytest.raises(InvalidInputError) as exc_info:
        engine.predict(inverted_negative)
    assert "negative" in str(exc_info.value).lower()

    # Biological range inverted vector (25.0 - normal) must be flagged as OOD
    inverted_bounded = [25.0] * len(engine.feature_names)
    result = engine.predict(inverted_bounded)
    assert result["is_ood"] is True or result["low_confidence"] is True


def test_negative_expression_values_rejected(engine):
    """Edge Case 4: Negative gene expression violates RNA-Seq non-negativity."""
    sample = [2.0] * len(engine.feature_names)
    sample[10] = -0.5  # Negative value
    with pytest.raises(InvalidInputError) as exc_info:
        engine.predict(sample)
    assert "negative" in str(exc_info.value).lower()


def test_extreme_unbiological_values_rejected(engine):
    """Edge Case 5: Values above biological maximum (25.0) must be rejected."""
    sample = [2.0] * len(engine.feature_names)
    sample[5] = 25.1
    with pytest.raises(InvalidInputError) as exc_info:
        engine.predict(sample)
    assert "biological maximum" in str(exc_info.value).lower() or "25" in str(exc_info.value)


def test_single_sample_batch_dataloader_safety():
    """Edge Case 6: DataLoader drop_last must prevent 1-sample batch BatchNorm crash."""
    n_features = 20
    n_samples = 33  # 33 % 32 = 1 (would cause batch size 1)
    batch_size = 32

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, size=n_samples)
    X_val = np.random.randn(10, n_features).astype(np.float32)
    y_val = np.random.randint(0, 2, size=10)

    model = CancerClassifierDNN(input_dim=n_features, num_classes=2, use_batch_norm=True)
    trainer = Trainer(model, device=torch.device("cpu"))
    train_loader, val_loader = trainer._make_loaders(X, y, X_val, y_val, batch_size=batch_size)

    # With drop_last=True for len > batch_size, the single trailing sample is dropped
    batches = list(train_loader)
    assert len(batches) == 1, "Expected only 1 full batch of 32 items, dropping remainder."
    assert batches[0][0].shape[0] == 32

    # Forward pass through BatchNorm1d must not crash
    model.train()
    out = model(batches[0][0])
    assert out.shape == (32, 2)


def test_temperature_scaling_calibrates_borderline_sample(engine):
    """Edge Case 7: Ambiguous Sample 46 confidence must soften below 0.75."""
    demo_file = ARTIFACTS_DIR / "demo_profiles.json"
    assert demo_file.exists(), "demo_profiles.json must exist."
    with demo_file.open("r", encoding="utf-8") as f:
        demos = json.load(f)

    sample46 = demos["sample46"]["gene_values"]

    # 1. Prediction with calibrated temperature (engine default: 3.0)
    calibrated_result = engine.predict(sample46)
    assert calibrated_result["confidence"] < CONFIDENCE_THRESHOLD, (
        f"Calibrated confidence {calibrated_result['confidence']} should be < {CONFIDENCE_THRESHOLD}"
    )
    assert calibrated_result["low_confidence"] is True
    assert "uncertain" in calibrated_result["warning"].lower()

    # 2. Raw uncalibrated prediction (temperature = 1.0)
    engine_unscaled = CancerInferenceEngine(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        encoder_path=ENCODER_PATH,
        features_path=FEATURES_PATH,
        temperature=1.0,
        device=torch.device("cpu"),
    )
    unscaled_result = engine_unscaled.predict(sample46)
    assert unscaled_result["confidence"] > 0.90, "Unscaled prediction exhibits softmax saturation (> 90%)."


def test_gradient_attribution_captures_negative_drivers(engine):
    """Edge Case 8: Input x Gradient must identify both positive drivers and suppressed tumor suppressors."""
    demo_file = ARTIFACTS_DIR / "demo_profiles.json"
    with demo_file.open("r", encoding="utf-8") as f:
        demos = json.load(f)

    result = engine.predict(demos["brca"]["gene_values"])

    # Top features (positive oncogenic drivers)
    assert len(result["top_features"]) > 0
    for feat in result["top_features"]:
        assert "attribution" in feat
        assert "expression" in feat

    # Suppressed features (negative attribution / tumor suppressors)
    assert "suppressed_features" in result
    for feat in result["suppressed_features"]:
        assert feat["attribution"] < 0, "Suppressed features must have negative attribution."


def test_raw_20531_features_ingestion_and_mapping(engine):
    """Edge Case 9: Ingestion of full 20,531 raw gene profile must map to 2,000 features."""
    raw_vector = [1.5] * 20531
    result = engine.predict(raw_vector)
    assert result["predicted_class"] in engine.class_names
    assert len(result["class_probabilities"]) == len(engine.class_names)

    # Also test dictionary mapping
    raw_dict = {f"gene_{i}": 2.0 for i in range(20531)}
    result_dict = engine.predict(raw_dict)
    assert result_dict["predicted_class"] in engine.class_names


def test_chat_fallback_clinical_richness():
    """Edge Case 10: Local fallback assistant produces rich clinical differential explanations."""
    mock_prediction = {
        "predicted_class": "BRCA",
        "confidence": 0.71,
        "class_probabilities": {"BRCA": 0.71, "LUAD": 0.25, "KIRC": 0.02, "COAD": 0.01, "PRAD": 0.01},
        "top_features": [
            {"gene": "gene_11457", "expression": 1.83, "attribution": 0.267},
            {"gene": "gene_14390", "expression": 2.34, "attribution": 0.255},
        ],
        "suppressed_features": [
            {"gene": "gene_8030", "expression": -3.17, "attribution": -0.476},
        ],
        "low_confidence": True,
        "warning": "Confidence 0.7100 is below threshold 0.75.",
    }

    # Query about tumor suppressors
    resp_suppressed = generate_local_fallback_response("What tumor suppressors are downregulated?", mock_prediction)
    assert "Suppressed Biomarkers" in resp_suppressed
    assert "gene_8030" in resp_suppressed

    # Query about biomarkers/drivers
    resp_drivers = generate_local_fallback_response("Which genes drive this prediction?", mock_prediction)
    assert "gene_11457" in resp_drivers
    assert "Activating Drivers" in resp_drivers

    # Query about confidence / margin
    resp_conf = generate_local_fallback_response("How confident is the model?", mock_prediction)
    assert "71.0%" in resp_conf
    assert "Clinician Alert" in resp_conf or "Low Confidence" in resp_conf


def test_pca_reducer_centroid_consistency(engine):
    """Edge Case 11: PCA reducer transforms reference samples into separated cohort centroids."""
    ref_path = ARTIFACTS_DIR / "pca_reference_points.json"
    assert ref_path.exists(), "pca_reference_points.json must exist."
    with ref_path.open("r", encoding="utf-8") as f:
        points = json.load(f)

    assert len(points) > 50, "Should have reference points."
    brca_points = [p for p in points if p["label"] == "BRCA"]
    kirc_points = [p for p in points if p["label"] == "KIRC"]

    brca_centroid = np.mean([[p["x"], p["y"]] for p in brca_points], axis=0)
    kirc_centroid = np.mean([[p["x"], p["y"]] for p in kirc_points], axis=0)

    distance = np.linalg.norm(brca_centroid - kirc_centroid)
    assert distance > 1.0, f"BRCA and KIRC centroids should be well separated in PCA space (got {distance:.2f})"


def test_hist_gradient_boosting_baseline():
    """Edge Case 12: HistGradientBoosting baseline must train without native library dependencies."""
    baselines = make_baselines(random_state=42)
    assert "hist_gradient_boosting" in baselines
    hgb = baselines["hist_gradient_boosting"]
    assert isinstance(hgb, HistGradientBoostingClassifier)

    # Train on synthetic data
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 50))
    y = (X[:, 0] > 0).astype(int)
    hgb.fit(X, y)
    preds = hgb.predict(X)
    acc = (preds == y).mean()
    assert acc > 0.90, f"HistGradientBoosting should fit separable data (got {acc:.2f})"


def test_demo_profiles_content(engine):
    """Edge Case 13: Curated demo profiles must contain valid BRCA, KIRC, and Sample 46."""
    demo_file = ARTIFACTS_DIR / "demo_profiles.json"
    with demo_file.open("r", encoding="utf-8") as f:
        demos = json.load(f)

    for key in ("brca", "kirc", "sample46"):
        assert key in demos, f"Key '{key}' missing from demo_profiles.json"
        profile = demos[key]
        assert "name" in profile
        assert "cohort" in profile
        assert len(profile["gene_values"]) == len(engine.feature_names)
        assert min(profile["gene_values"]) >= MIN_EXPRESSION_VALUE
        assert max(profile["gene_values"]) <= MAX_EXPRESSION_VALUE


def test_empty_input_vector_rejected(engine):
    """Edge Case 14: Empty input vector must raise InvalidInputError."""
    with pytest.raises(InvalidInputError) as exc_info:
        engine.predict([])
    assert "empty" in str(exc_info.value).lower()


def test_invalid_feature_length_rejected(engine):
    """Edge Case 15: Invalid feature vector length must raise InvalidInputError."""
    with pytest.raises(InvalidInputError) as exc_info:
        engine.predict([1.0, 2.0, 3.0])
    assert f"expected {len(engine.feature_names)}" in str(exc_info.value).lower()


def test_api_predict_accepts_raw_20531_vector():
    """Edge Case 16: API POST /predict must accept and map full 20,531 raw feature vector."""
    client = TestClient(app)
    raw_vector = [2.5] * 20531
    response = client.post("/predict", json={"sample_id": "raw_patient_001", "gene_values": raw_vector})
    assert response.status_code == 200, f"Failed with {response.status_code}: {response.text}"
    data = response.json()
    assert "predicted_class" in data
    assert len(data["class_probabilities"]) == 5


def test_api_predict_accepts_dictionary():
    """Edge Case 17: API POST /predict must accept dictionary of gene expression values."""
    client = TestClient(app)
    gene_dict = {f"gene_{i}": 3.0 for i in range(100)}
    response = client.post("/predict", json={"sample_id": "dict_patient_001", "gene_values": gene_dict})
    assert response.status_code == 200, f"Failed with {response.status_code}: {response.text}"
    data = response.json()
    assert "predicted_class" in data


def test_inference_does_not_accumulate_model_parameter_gradients(engine):
    """Edge Case 18: Input x Gradient attribution must NOT leak or accumulate gradients on model weights."""
    sample = [2.0] * len(engine.feature_names)
    
    # Zero out any existing grads
    engine.model.zero_grad(set_to_none=True)
    
    # Run multiple inference iterations
    for _ in range(3):
        engine.predict(sample)

    for name, param in engine.model.named_parameters():
        assert param.grad is None or float(param.grad.norm()) == 0.0, (
            f"Parameter {name} accumulated gradient during inference!"
        )


def test_sample_46_not_flagged_as_ood(engine):
    """Edge Case 19: Ambiguous Sample 46 is a real patient biopsy and must NOT be flagged as OOD."""
    demo_file = ARTIFACTS_DIR / "demo_profiles.json"
    with demo_file.open("r", encoding="utf-8") as f:
        demos = json.load(f)

    result = engine.predict(demos["sample46"]["gene_values"])
    assert result["is_ood"] is False, "Sample 46 is a genuine biological sample, not synthetic OOD noise."
    assert result["low_confidence"] is True, "Sample 46 must trigger low confidence alert."
    assert "out-of-distribution" not in (result["warning"] or "").lower()


def test_api_demo_profiles_endpoint():
    """Edge Case 20: GET /demo_profiles returns curated profiles for frontend."""
    client = TestClient(app)
    response = client.get("/demo_profiles")
    assert response.status_code == 200
    data = response.json()
    assert "brca" in data
    assert "kirc" in data
    assert "sample46" in data


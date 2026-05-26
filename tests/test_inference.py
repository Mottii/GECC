import json

import joblib
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.inference import CancerInferenceEngine
from src.models.deep_net import CancerClassifierDNN


def test_inference_engine_predicts_from_temp_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("app.inference.HIDDEN_LAYERS", [6])
    monkeypatch.setattr("app.inference.DROPOUT_RATE", 0.0)

    features = ["gene_a", "gene_b", "gene_c", "gene_d"]
    classes = np.array(["BRCA", "KIRC", "LUAD"])

    scaler = StandardScaler().fit(np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.float32))
    encoder = LabelEncoder().fit(classes)
    model = CancerClassifierDNN(input_dim=len(features), hidden_layers=[6], num_classes=len(classes), dropout_rate=0.0)

    model_path = tmp_path / "model.pth"
    scaler_path = tmp_path / "scaler.pkl"
    encoder_path = tmp_path / "encoder.pkl"
    features_path = tmp_path / "features.json"
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)
    features_path.write_text(json.dumps(features), encoding="utf-8")

    engine = CancerInferenceEngine(
        model_path=model_path,
        scaler_path=scaler_path,
        encoder_path=encoder_path,
        features_path=features_path,
        device=torch.device("cpu"),
    )

    result = engine.predict([0.5, 1.5, 2.5, 3.5])

    assert engine.model_loaded is True
    assert result["predicted_class"] in classes
    assert set(result["class_probabilities"]) == set(classes)
    assert len(result["top_features"]) == 4
    assert "low_confidence" in result
    assert "warning" in result
    assert result["model_version"].startswith("file-")

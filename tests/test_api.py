from fastapi.testclient import TestClient

from app import main
from app.inference import InvalidInputError


class FakeEngine:
    model_loaded = True
    class_names = ["BRCA", "KIRC"]
    feature_names = ["gene_a", "gene_b", "gene_c"]

    def readiness_message(self):
        return "ready"

    def reload(self):
        return None

    def model_metadata(self):
        return {"model_name": "CancerClassifierDNN", "model_version": "test-run"}

    def predict(self, gene_values):
        return {
            "predicted_class": "BRCA",
            "confidence": 0.8,
            "class_probabilities": {"BRCA": 0.8, "KIRC": 0.2},
            "top_features": [{"gene": "gene_a", "expression": 1.0}],
            "low_confidence": False,
            "warning": None,
            "model_version": "test-run",
            "pca_coords": {"x": 1.2, "y": -0.8},
        }


class FakeEngineInvalid(FakeEngine):
    def predict(self, gene_values):
        raise InvalidInputError("gene_values must not contain NaN or infinite values.")


def test_api_health_features_and_predict(monkeypatch):
    monkeypatch.setattr(main, "engine", FakeEngine())
    client = TestClient(main.app)

    assert client.get("/health").json()["model_loaded"] is True
    assert "model_metadata" in client.get("/health").json()
    assert client.get("/classes").json()["classes"] == ["BRCA", "KIRC"]
    assert client.get("/features").json()["count"] == 3

    response = client.post("/predict", json={"sample_id": "S1", "gene_values": [1, 2, 3]})
    assert response.status_code == 200
    body = response.json()
    assert body["sample_id"] == "S1"
    assert body["predicted_class"] == "BRCA"
    assert body["model_version"] == "test-run"
    assert body["low_confidence"] is False


def test_api_validates_feature_length(monkeypatch):
    monkeypatch.setattr(main, "engine", FakeEngine())
    client = TestClient(main.app)

    response = client.post("/predict", json={"sample_id": "S1", "gene_values": [1, 2]})

    assert response.status_code == 422
    assert "Expected 3 features" in response.json()["detail"]


def test_api_rejects_invalid_values(monkeypatch):
    monkeypatch.setattr(main, "engine", FakeEngineInvalid())
    client = TestClient(main.app)
    response = client.post("/predict", json={"sample_id": "S1", "gene_values": [1, 2, 3]})
    assert response.status_code == 422
    assert "must not contain NaN" in response.json()["detail"]


def test_api_pca_reference_and_chat(monkeypatch):
    monkeypatch.setattr(main, "engine", FakeEngine())
    client = TestClient(main.app)

    # Test /pca_reference
    pca_response = client.get("/pca_reference")
    assert pca_response.status_code == 200
    assert isinstance(pca_response.json(), list)

    # Test /chat
    chat_payload = {
        "message": "Explain BRCA result",
        "history": [],
        "prediction": {
            "predicted_class": "BRCA",
            "confidence": 0.8,
            "class_probabilities": {"BRCA": 0.8, "KIRC": 0.2},
            "top_features": [{"gene": "gene_a", "expression": 1.0}],
            "low_confidence": False
        }
    }
    chat_response = client.post("/chat", json=chat_payload)
    assert chat_response.status_code == 200
    assert "reply" in chat_response.json()
    assert isinstance(chat_response.json()["reply"], str)

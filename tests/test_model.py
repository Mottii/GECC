import torch

from src.models.deep_net import CancerClassifierDNN


def test_dnn_forward_shape_and_probabilities():
    model = CancerClassifierDNN(input_dim=12, hidden_layers=[8, 4], num_classes=5, dropout_rate=0.0)
    model.eval()
    x = torch.randn(3, 12)

    logits = model(x)
    probabilities = model.predict_proba(x)

    assert logits.shape == (3, 5)
    assert probabilities.shape == (3, 5)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3), atol=1e-6)

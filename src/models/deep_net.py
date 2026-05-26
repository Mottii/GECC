import torch
import torch.nn as nn

from src.utils.config import BATCH_NORM, DROPOUT_RATE, HIDDEN_LAYERS, NUM_CLASSES, TOP_K_FEATURES


class CancerClassifierDNN(nn.Module):
    """Feed-forward DNN for RNA-Seq cancer classification."""

    def __init__(
        self,
        input_dim: int = TOP_K_FEATURES,
        hidden_layers: list[int] | None = None,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = DROPOUT_RATE,
        use_batch_norm: bool = BATCH_NORM,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        hidden_layers = hidden_layers or HIDDEN_LAYERS

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)

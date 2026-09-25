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
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.temperature = float(temperature)
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

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(float(temperature), 0.01)

    def calibrate_temperature(
        self,
        val_inputs: torch.Tensor,
        val_targets: torch.Tensor | np.ndarray,
        lr: float = 0.01,
        max_iter: int = 50,
        device: torch.device | None = None,
    ) -> float:
        """Fit a post-hoc temperature scalar T to optimize negative log-likelihood on validation logits.
        
        Guarantees that softmax probabilities are calibrated to reflect true empirical uncertainty.
        """
        dev = device or next(self.parameters()).device
        self.eval()

        x_val = val_inputs.to(dev)
        y_val = torch.as_tensor(val_targets, dtype=torch.long, device=dev)
        with torch.no_grad():
            logits = self.forward(x_val)

        temp_param = nn.Parameter(torch.ones(1, device=dev) * 1.5)
        optimizer = torch.optim.LBFGS([temp_param], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def _step():
            optimizer.zero_grad()
            t = torch.clamp(temp_param, min=0.1, max=10.0)
            loss = criterion(logits / t, y_val)
            loss.backward()
            return loss

        optimizer.step(_step)
        fitted_t = float(torch.clamp(temp_param, min=0.1, max=10.0).item())
        optimal_t = max(round(fitted_t, 3), 0.1)
        self.set_temperature(optimal_t)
        return optimal_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor, temperature: float | None = None) -> torch.Tensor:
        temp = temperature if temperature is not None else getattr(self, "temperature", 1.0)
        temp = max(float(temp), 0.01)
        return torch.softmax(self.forward(x) / temp, dim=-1)

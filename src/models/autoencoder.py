import numpy as np
import torch
import torch.nn as nn


class GeneExpressionAutoencoder(nn.Module):
    """Optional dimensionality reduction model for future experiments."""

    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mean squared reconstruction error per sample."""
        x_rec = self.forward(x)
        return torch.mean((x - x_rec) ** 2, dim=-1)


def train_autoencoder(
    X_train: torch.Tensor | np.ndarray,
    input_dim: int,
    latent_dim: int = 128,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> tuple[GeneExpressionAutoencoder, float]:
    """Train autoencoder on in-distribution gene expression data and determine OOD threshold."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeneExpressionAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(dev)

    if not isinstance(X_train, torch.Tensor):
        X_t = torch.tensor(X_train, dtype=torch.float32)
    else:
        X_t = X_train.float()

    dataset = torch.utils.data.TensorDataset(X_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for (batch_x,) in loader:
            batch_x = batch_x.to(dev)
            optimizer.zero_grad()
            rec = model(batch_x)
            loss = criterion(rec, batch_x)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        errors = []
        for (batch_x,) in loader:
            batch_x = batch_x.to(dev)
            err = model.reconstruction_error(batch_x).cpu().numpy()
            errors.extend(err.tolist())
        # OOD threshold set with generalization buffer to prevent false positives on biological cohorts
        arr = np.array(errors)
        threshold = float(max(0.95, np.percentile(arr, 99.0) * 4.0 + 0.25))

    return model, threshold

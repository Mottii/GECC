from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from src.training.callbacks import EarlyStoppingState
from src.utils.config import (
    BATCH_SIZE,
    EARLY_STOPPING,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    WEIGHT_DECAY,
)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        model_dir: Path = MODELS_DIR,
        seed: int | None = None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.seed = seed
        self.model.to(self.device)

    def _make_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = BATCH_SIZE,
    ) -> tuple[DataLoader, DataLoader]:
        def to_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
            return TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
            )

        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        drop_last = len(X_train) > batch_size
        train_loader = DataLoader(
            to_dataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            drop_last=drop_last,
        )
        val_loader = DataLoader(to_dataset(X_val, y_val), batch_size=batch_size)
        return train_loader, val_loader

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
    ) -> dict[str, list[float]]:
        train_loader, val_loader = self._make_loaders(X_train, y_train, X_val, y_val, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        best_val_loss = float("inf")
        best_val_acc = 0.0
        early_stopper = EarlyStoppingState(patience=EARLY_STOPPING, mode="min")
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        epochs_ran = 0

        for epoch in range(1, epochs + 1):
            epochs_ran = epoch
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            val_loss, val_acc = self.evaluate_loader(val_loader, criterion)
            scheduler.step()

            avg_train_loss = train_loss / max(1, len(train_loader))
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(
                f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} "
                f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                self.model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self.model_dir / "deep_net_best.pth")
                print(f"  New best model saved (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")

            if early_stopper.step(val_loss):
                print(f"Early stopping at epoch {epoch} (val_loss ceased improving)")
                break

        history["best_val_loss"] = [best_val_loss]
        history["best_val_acc"] = [best_val_acc]
        history["epochs_ran"] = [epochs_ran]
        return history

    def evaluate_loader(self, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.model(X_batch)
                total_loss += criterion(logits, y_batch).item()
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        return total_loss / max(1, len(loader)), correct / max(1, total)

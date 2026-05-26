from dataclasses import dataclass


@dataclass
class EarlyStoppingState:
    patience: int
    best_score: float = float("-inf")
    epochs_without_improvement: int = 0

    def step(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

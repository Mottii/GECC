from dataclasses import dataclass


@dataclass
class EarlyStoppingState:
    patience: int
    mode: str = "min"
    best_score: float = float("inf")
    epochs_without_improvement: int = 0

    def __post_init__(self):
        if self.mode == "max" and self.best_score == float("inf"):
            self.best_score = float("-inf")
        elif self.mode == "min" and self.best_score == float("-inf"):
            self.best_score = float("inf")

    def step(self, score: float) -> bool:
        improved = (score < self.best_score) if self.mode == "min" else (score > self.best_score)
        if improved:
            self.best_score = score
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

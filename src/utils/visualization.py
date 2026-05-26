from pathlib import Path

import matplotlib.pyplot as plt


def save_or_show(save_path: str | Path | None = None, show: bool = False) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

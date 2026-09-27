"""Plot saved training and validation accuracy against epoch."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    project_dir = Path(__file__).resolve().parent
    history_path = project_dir / "artifacts" / "history.json"
    if not history_path.is_file():
        raise SystemExit("Training history is missing. Run python train.py first.")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    train_accuracy = history["train_metrics"]
    valid_accuracy = history["valid_metrics"]
    if not train_accuracy or len(train_accuracy) != len(valid_accuracy):
        raise ValueError("Training and validation accuracy must have equal, nonzero lengths.")
    epochs = range(1, len(train_accuracy) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_accuracy, "o-", label="Training accuracy")
    ax.plot(epochs, valid_accuracy, "s-", label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("FashionMNIST training and validation accuracy")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = project_dir / "training_accuracy.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved accuracy plot to {output_path}")


if __name__ == "__main__":
    main()

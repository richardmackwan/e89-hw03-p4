"""Train FashionMNIST for 20 epochs and save the history and model weights."""

import json
from pathlib import Path

import torch
from torchmetrics import Accuracy

from load_data import load_data
from model import ImageClassifier, loss_fn
from setup import device
from train_utils import train2


def main():
    torch.manual_seed(42)
    print(f"Using device: {device}")
    train_loader, valid_loader, _ = load_data()
    model = ImageClassifier().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accuracy = Accuracy(task="multiclass", num_classes=10, average="micro").to(device)

    history = train2(
        model, optimizer, loss_fn, accuracy, train_loader, valid_loader,
        n_epochs=20,
    )

    output_dir = Path(__file__).resolve().parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    # Save CPU weights so the model can also be loaded without a GPU.
    model_path = output_dir / "model.pt"
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        model_path,
    )
    print(f"Saved training history to {history_path}")
    print(f"Saved model weights to {model_path}")


if __name__ == "__main__":
    main()

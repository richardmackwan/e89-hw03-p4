"""Display predictions and probabilities for the first three validation images."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from load_data import load_data
from model import ImageClassifier

# __file__ is undefined when this code runs as a notebook cell.
PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def main():
    model_path = PROJECT_DIR / "artifacts" / "model.pt"
    if not model_path.is_file():
        raise SystemExit("Trained weights are missing. Run python train.py first.")

    model = ImageClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    _, valid_loader, _ = load_data()
    valid_data = valid_loader.dataset
    class_names = valid_data.dataset.classes
    samples = [valid_data[index] for index in range(3)]
    images = torch.stack([image for image, _ in samples])

    with torch.inference_mode():
        probabilities = torch.softmax(model(images), dim=1)
        predictions = probabilities.argmax(dim=1)
        top_probs, top_classes = probabilities.topk(4, dim=1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for index, (ax, (image, true_label)) in enumerate(zip(axes, samples)):
        predicted_name = class_names[predictions[index].item()]
        true_name = class_names[true_label]
        print(f"\nValidation image {index + 1}")
        print(f"Predicted: {predicted_name} | True: {true_name}")
        print("Softmax probabilities:")
        for name, probability in zip(class_names, probabilities[index].tolist()):
            print(f"  {name}: {probability:.6f}")
        print("Top-4 probabilities (highest first):")
        for class_index, probability in zip(
            top_classes[index].tolist(), top_probs[index].tolist()
        ):
            print(f"  {class_names[class_index]}: {probability:.6f}")

        ax.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Predicted: {predicted_name}\nTrue: {true_name}")
        ax.axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

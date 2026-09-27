"""Print a training sample's details and display labeled FashionMNIST images."""

import matplotlib.pyplot as plt

from load_data import load_data


def main():
    train_loader, _, _ = load_data()
    train_data = train_loader.dataset
    # random_split returns a Subset; class names belong to its source dataset.
    class_names = train_data.dataset.classes

    image, label = train_data[0]
    print(f"Shape: {tuple(image.shape)}")
    print(f"Dtype: {image.dtype}")
    print(f"Class name: {class_names[label]}")

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for index, ax in enumerate(axes.flat):
        image, label = train_data[index]
        ax.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(class_names[label])
        ax.axis("off")

    fig.suptitle("FashionMNIST training samples")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

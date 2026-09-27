"""Define the FashionMNIST multilayer perceptron and classification loss."""

import torch
from torch import nn


torch.manual_seed(42)


class ImageClassifier(nn.Module):
    """Map 28-by-28 grayscale images to logits for ten classes."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        return self.layers(x)


loss_fn = nn.CrossEntropyLoss()

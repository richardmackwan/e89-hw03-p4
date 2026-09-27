"""Step 1: Import the libraries and select the compute device."""

import torch
from torch import nn
import torchvision
import torchmetrics
import matplotlib.pyplot as plt

# Prefer an NVIDIA GPU, then an Apple GPU, then the CPU.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

if __name__ == "__main__":
    print(f"Using device: {device}")

"""
This script provides utility functions for setting random seeds for reproducibility and
plotting training and validation losses over epochs.

Author: yumemonzo@gmail.com
Date: 2025-02-21
"""

import random
import torch
import matplotlib.pyplot as plt
from typing import List, Optional


def seed_everything(seed: int = 42) -> None:
    """
    Set seed for reproducibility across different modules.

    Args:
        seed (int, optional): The seed value to use. Defaults to 42.
    """
    random.seed(seed)  # Python built-in random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed

    torch.backends.cudnn.deterministic = True  # Ensures deterministic execution
    torch.backends.cudnn.benchmark = False  # Disable if model structure is not fixed


import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

def plot_losses(
    title: str, 
    train_losses: List[float], 
    valid_losses: List[Optional[float]],
    epochs: int,
    save_path: str
) -> None:
    """
    Plot training and validation loss over epochs.

    Args:
        title (str): Title of the plot.
        train_losses (List[float]): Training loss per epoch.
        valid_losses (List[Optional[float]]): Validation loss per epoch (may contain None).
        epochs (int): Number of epochs.
        valid_interval (int): Number of validation steps.
        save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    epochs_range = list(range(1, epochs + 1))
    
    plt.plot(epochs_range, train_losses, label="Train Loss", linestyle='-')

    # Validation Loss에서 None 값 제외하고 유효한 X축, Y축 데이터 생성
    valid_x = [epoch for epoch, loss in zip(epochs_range, valid_losses) if loss is not None]
    valid_y = [loss for loss in valid_losses if loss is not None]

    plt.plot(valid_x, valid_y, label="Validation Loss", linestyle='--')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title)
    plt.grid(True)

    plt.savefig(save_path)
    plt.clf()
    
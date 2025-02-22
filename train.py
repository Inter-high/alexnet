"""
This script serves as the main entry point for training and evaluating the AlexNet model on the CIFAR-10 dataset.
It loads the dataset, splits it, initializes the model, and executes training and evaluation.
Logging, seed initialization, and loss plotting are also integrated.

Author: yumemonzo@gmail.com
Date: 2025-02-21
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import hydra
from omegaconf import DictConfig
from data import get_datasets, split_dataset, get_train_loader, get_test_loader
from model import AlexNet
from trainer import Trainer
from utils import seed_everything, plot_losses
from torchvision.models import resnet18


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    """
    Main function to train and evaluate models.
    
    Args:
        cfg (DictConfig): Configuration parameters from Hydra.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set seed for reproducibility
    seed_everything(cfg['seed'])

    # Load and split datasets
    dataset, test_dataset = get_datasets(cfg['data']['data_dir'])
    train_dataset, valid_dataset = split_dataset(cfg['seed'], dataset)

    # Create data loaders
    train_loader, valid_loader = get_train_loader(train_dataset, valid_dataset, cfg['data']['batch_size'], cfg['data']['num_workers'])
    test_loader = get_test_loader(test_dataset, cfg['data']['test_batch_size'], cfg['data']['num_workers'])

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model and training setup
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    logger.info(f"Model initialized: {model}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), cfg['train']['lr'], momentum=0.9, weight_decay=0.0005)

    # Define path for saving model weights
    weight_path = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        "best_model.pth"
    )

    trainer = Trainer(model, optimizer, criterion, device, logger, hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    logger.info("========== Training Start ==========")
    train_losses, valid_losses = trainer.training(train_loader, valid_loader, cfg['train']['epochs'], cfg['train']['valid_interval'], weight_path)

    # Save loss plot
    plot_losses("CIFAR-10 ResNet18", train_losses, valid_losses, cfg['train']['epochs'],
                os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "losses.jpg"))

    logger.info("========== Training End ==========")

    logger.info("========== Evaluation Start ==========")
    top1_error, top5_error = trainer.test(test_loader)
    logger.info(f"Top-1 Error: {top1_error:.4f} | Top-5 Error: {top5_error:.4f}")
    logger.info("========== Evaluation End ==========")

    # Save training results as a pickle file
    pickle_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "training_results.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "top1_error": top1_error,
            "top5_error": top5_error
        }, f)

    logger.info(f"Training results saved to {pickle_path}")


if __name__ == "__main__":
    my_app()

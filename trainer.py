"""
This script defines a Trainer class for training, validating, and testing deep learning models.
It includes training with learning rate scheduling, validation, and testing with top-1 and top-5 accuracy computation.
TensorBoard logging is also integrated for tracking loss progression.

Author: yumemonzo@gmail.com
Date: 2025-02-21
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import List, Tuple, Optional


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str,
        logger,
        log_dir: str = "./logs",
    ) -> None:
        """
        Initializes the Trainer class.

        Args:
            model (nn.Module): Neural network model to train.
            optimizer (optim.Optimizer): Optimizer for training.
            criterion (nn.Module): Loss function.
            device (str): Device to run training on ('cpu' or 'cuda').
            logger: Logger for logging information.
            log_dir (str): Directory for TensorBoard logs. Defaults to './logs'.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.lowest_loss = float("inf")
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self, train_dataloader: torch.utils.data.DataLoader) -> float:
        """
        Executes one epoch of training.

        Args:
            train_dataloader (DataLoader): Dataloader for training data.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc="Training", leave=True)
        
        for x, y in progress_bar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        return total_loss / len(train_dataloader)

    def valid(self, valid_dataloader: torch.utils.data.DataLoader) -> float:
        """
        Executes one epoch of validation.

        Args:
            valid_dataloader (DataLoader): Dataloader for validation data.

        Returns:
            float: Average validation loss for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(valid_dataloader, desc="Validating", leave=True)
        
        with torch.no_grad():
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y).item()
                total_loss += loss
                progress_bar.set_postfix(loss=loss)
        
        return total_loss / len(valid_dataloader)

    def test(self, test_dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Evaluates the model on the test dataset.

        Args:
            test_dataloader (DataLoader): Dataloader for test data.

        Returns:
            Tuple[float, float]: Top-1 and top-5 error rates.
        """
        self.model.eval()
        top1_correct = 0
        top5_correct = 0
        total_samples = 0
        progress_bar = tqdm(test_dataloader, desc="Testing", leave=True)
        
        with torch.no_grad():
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                _, top5_preds = torch.topk(outputs, k=5, dim=1)
                top1_preds = top5_preds[:, 0]
                total_samples += y.size(0)
                top1_correct += (top1_preds == y).sum().item()
                top5_correct += sum(y[i].item() in top5_preds[i].tolist() for i in range(y.size(0)))
                progress_bar.set_postfix(top1_acc=top1_correct / total_samples, top5_acc=top5_correct / total_samples)
        
        top1_error = 1 - (top1_correct / total_samples)
        top5_error = 1 - (top5_correct / total_samples)
        
        return top1_error, top5_error

    def training(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        epochs: int,
        valid_interval: int,
        weight_path: str,
    ) -> Tuple[List[float], List[Optional[float]]]:
        """
        Trains the model for a specified number of epochs with validation and learning rate adjustments.

        Args:
            train_dataloader (DataLoader): Training dataset loader.
            valid_dataloader (DataLoader): Validation dataset loader.
            epochs (int): Number of training epochs.
            valid_interval (int): Number of epochs between validations.
            weight_path (str): Path to save the best model weights.

        Returns:
            Tuple[List[float], List[Optional[float]]]: Training and validation loss history.
        """
        train_losses = []
        valid_losses = [None] * epochs
        initial_lr = self.optimizer.param_groups[0]['lr']
        lr = initial_lr
        patience = 3
        decrease_count = 0
        no_improve_count = 0
        validation_interval = max(1, epochs // valid_interval)
        
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            train_loss = self.train(train_dataloader)
            train_losses.append(train_loss)
            
            valid_loss = None
            if epoch % validation_interval == 0:
                valid_loss = self.valid(valid_dataloader)
                valid_losses[epoch - 1] = valid_loss

                if valid_loss < self.lowest_loss:
                    self.lowest_loss = valid_loss
                    torch.save(self.model.state_dict(), weight_path)
                    self.logger.info(f"Epoch: {epoch}/{epochs} | New best model saved with Valid Loss: {valid_loss:.4f}")
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if no_improve_count >= 3 and decrease_count < patience:
                    lr /= 10
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    self.logger.info(f"Epoch: {epoch}/{epochs} | Learning rate decreased to {lr}")
                    decrease_count += 1
                    no_improve_count = 0
                
                self.logger.info(f"Epoch: {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | LR: {lr:.6f}")
            
            # TensorBoard 기록
            loss_dict = {"Train": train_loss}
            if valid_loss is not None:
                loss_dict["Validation"] = valid_loss
            
            self.writer.add_scalars("Loss", loss_dict, epoch)
        
        self.writer.close()
        return train_losses, valid_losses

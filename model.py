"""
This script defines the AlexNet architecture using PyTorch.
The model consists of convolutional layers followed by fully connected layers.
Local response normalization (LRN) is used after the first two convolutional layers.
Dropout is applied to the fully connected layers to prevent overfitting.

Author: yumemonzo@gmail.com
Date: 2025-02-21
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        """
        Initializes the AlexNet model.

        Args:
            num_classes (int): Number of output classes. Default is 1000.
        """
        super().__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc_layer1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.fc_layer3 = nn.Sequential(
            nn.Linear(4096, num_classes),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m in [self.conv_layer2[0], self.conv_layer4[0], self.conv_layer5[0]]:
                    nn.init.constant_(m.bias, 1.0)
                else:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 3, 224, 224).
        
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, num_classes).
        """
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)

        return x
    
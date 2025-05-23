"""
Attack models for attribute inference.
"""
from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class AttackClassifier(nn.Module):
    """
    Neural network for classifying the target attribute.
    Used for categorical attribute inference attacks.
    """
    def __init__(self, input_size: int, hidden_size: int = 32, num_classes: int = 2) -> None:
        """
        Initialize the attack classifier.
        
        Args:
            input_size: Size of the input features
            hidden_size: Size of the hidden layer
            num_classes: Number of classes to predict
        """
        super(AttackClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after forward pass
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x


class AttackRegressor(nn.Module):
    """
    Neural network for regressing the target attribute.
    Used for continuous attribute inference attacks.
    """
    def __init__(self, input_size: int, hidden_size: int = 32, output_size: int = 1) -> None:
        """
        Initialize the attack regressor.
        
        Args:
            input_size: Size of the input features
            hidden_size: Size of the hidden layer
            output_size: Size of the output (typically 1 for regression)
        """
        super(AttackRegressor, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after forward pass
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

"""
Neural network models for classification tasks.
"""
from typing import Optional, Union, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset


class NurseryClassifier(nn.Module):
    """
    Neural network classifier for the Nursery dataset.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_classes: int = 5) -> None:
        """
        Initialize the classifier model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of the hidden layer
            num_classes: Number of output classes
        """
        super(NurseryClassifier, self).__init__()
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
    
    def predict_proba(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Method to match scikit-learn's API for probability prediction.
        
        Args:
            x: Input data (numpy array or torch tensor)
            
        Returns:
            Predicted probabilities
        """
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            return self.forward(x).numpy()


class ModelTrainer:
    """
    Trainer class for neural network models.
    """
    def __init__(self, model: nn.Module, criterion: Optional[nn.Module] = None, 
                 optimizer: Optional[optim.Optimizer] = None, lr: float = 0.001) -> None:
        """
        Initialize the model trainer.
        
        Args:
            model: PyTorch model to train
            criterion: Loss function (default: BCELoss)
            optimizer: Optimizer (default: Adam)
            lr: Learning rate for the optimizer
        """
        self.model = model
        self.criterion = criterion or nn.BCELoss()
        
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
            
    def train(self, train_dataset: Dataset, num_epochs: int = 10, 
              batch_size: int = 64, verbose: bool = True) -> nn.Module:
        """
        Train the model.
        
        Args:
            train_dataset: PyTorch dataset containing training data
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print training progress
            
        Returns:
            Trained model
        """
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, labels in train_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if verbose and (epoch+1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
        
        return self.model
    
    def evaluate(self, test_dataset: Dataset, batch_size: int = 64) -> float:
        """
        Evaluate the model.
        
        Args:
            test_dataset: PyTorch dataset containing test data
            batch_size: Batch size for evaluation
            
        Returns:
            Model accuracy
        """
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                _, labels_max = torch.max(labels.data, 1)
                total += labels.size(0)
                correct += (predicted == labels_max).sum().item()
        
        accuracy = correct / total
        return accuracy

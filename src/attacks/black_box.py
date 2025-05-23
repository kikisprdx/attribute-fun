"""
Black-box attribute inference attack implementation.
"""
from typing import List, Optional, Tuple, Union, Any, Type

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from src.attacks.models import AttackClassifier, AttackRegressor


class AttributeInferenceBlackBox:
    """
    Black-box attribute inference attack.
    
    This attack assumes access only to the target model's predictions,
    not its internal parameters or architecture.
    """
    
    def __init__(self, target_model: Any, attack_feature: int) -> None:
        """
        Initialize the black-box attack.
        
        Args:
            target_model: The target model to attack
            attack_feature: Index of the feature to be attacked
        """
        self.target_model = target_model
        self.attack_feature = attack_feature
        self.attack_model: Optional[Union[AttackClassifier, AttackRegressor]] = None
        self.unique_values: Optional[np.ndarray] = None
        self.is_categorical: bool = True
        
    def fit(self, x: np.ndarray, epochs: int = 50, batch_size: int = 32) -> 'AttributeInferenceBlackBox':
        """
        Train the attack model.
        
        Args:
            x: Training data including the attack feature
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Self instance for method chaining
        """
        target_feature = x[:, self.attack_feature]
        
        self.unique_values = np.unique(target_feature)
        print(f"Unique values in training: {self.unique_values}")
        
        x_tensor = torch.FloatTensor(x)
        with torch.no_grad():
            predictions = self.target_model(x_tensor).numpy()
        
        x_without_feature = np.delete(x, self.attack_feature, axis=1)
        
        attack_input = np.concatenate((x_without_feature, predictions), axis=1)
        
        self.is_categorical = len(self.unique_values) <= 10
        
        if self.is_categorical:
            attack_feature_classes = np.zeros(len(target_feature), dtype=int)
            for i, val in enumerate(self.unique_values):
                attack_feature_classes[target_feature == val] = i
            
            print(f"Mapped target indices range: [{np.min(attack_feature_classes)}, {np.max(attack_feature_classes)}]")
            
            self.attack_model = self._train_classifier(
                torch.FloatTensor(attack_input), 
                torch.LongTensor(attack_feature_classes),
                epochs=epochs,
                batch_size=batch_size
            )
        else:
            self.attack_model = self._train_regressor(
                torch.FloatTensor(attack_input), 
                torch.FloatTensor(target_feature.reshape(-1, 1)),
                epochs=epochs,
                batch_size=batch_size
            )
            
        return self
        
    def _train_classifier(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, 
                          epochs: int = 50, batch_size: int = 32) -> AttackClassifier:
        """
        Train the attack classifier model.
        
        Args:
            x_tensor: Input tensor
            y_tensor: Target tensor
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Trained classifier model
        """
        print(f"Target tensor min: {y_tensor.min()}, max: {y_tensor.max()}")
        
        if y_tensor.min() < 0:
            raise ValueError(f"Negative target values detected: min={y_tensor.min()}")
            
        num_classes = y_tensor.max().item() + 1
        
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        input_size = x_tensor.shape[1]
        model = AttackClassifier(input_size=input_size, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}')
                
        return model
    
    def _train_regressor(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, 
                         epochs: int = 50, batch_size: int = 32) -> AttackRegressor:
        """
        Train the attack regressor model.
        
        Args:
            x_tensor: Input tensor
            y_tensor: Target tensor
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Trained regressor model
        """
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        input_size = x_tensor.shape[1]
        model = AttackRegressor(input_size=input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}')
                
        return model
    
    def infer(self, x: np.ndarray, pred: Optional[np.ndarray] = None, 
              values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Infer the attack feature values.
        
        Args:
            x: Data without the attack feature
            pred: Predictions from the target model (optional)
            values: Possible values of the attack feature
            
        Returns:
            Inferred values of the attack feature
        """
        if self.attack_model is None:
            raise ValueError("Attack model not trained. Call fit() first.")
            
        if pred is None:
            raise ValueError("Predictions must be provided when the feature is removed")
            
        attack_test_data = np.concatenate((x, pred), axis=1)
        attack_test_tensor = torch.FloatTensor(attack_test_data)
        
        with torch.no_grad():
            raw_predictions = self.attack_model(attack_test_tensor).numpy()
        
        if self.is_categorical:
            class_indices = np.argmax(raw_predictions, axis=1)
            
            inferred_values = np.zeros(len(class_indices))
            for i, idx in enumerate(class_indices):
                inferred_values[i] = self.unique_values[idx]
        else:
            inferred_values = raw_predictions.ravel()
        
        return inferred_values.reshape(1, -1)

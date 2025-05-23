"""
Data loading and preprocessing utilities for attribute inference attacks.
"""
from typing import Tuple, List, Optional, Dict, Any, Union

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve


class NurseryDataLoader:
    """
    Loader for the Nursery dataset with preprocessing for attribute inference attacks.
    """
    
    def __init__(self, data_path: Optional[str] = None, test_size: float = 0.5, 
                 transform_social: bool = True, random_state: int = 42) -> None:
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the dataset. If None, will download from UCI
            test_size: Proportion of the dataset to include in the test split
            transform_social: If True, transform the social feature to binary (0,1)
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path or 'nursery.data'
        self.test_size = test_size
        self.transform_social = transform_social
        self.random_state = random_state
        
        self.feature_names: Optional[np.ndarray] = None
        self.social_indices: Optional[List[int]] = None
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.x_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.train_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None
        
    def _download_if_needed(self) -> None:
        """Download the dataset if it doesn't exist locally."""
        if not os.path.exists(self.data_path):
            print(f"Downloading nursery dataset to {self.data_path}...")
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data'
            urlretrieve(url, self.data_path)
            print("Download complete.")
            
    def load(self) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                          Tuple[np.ndarray, np.ndarray], 
                          np.ndarray, 
                          List[int], 
                          Tuple[TensorDataset, TensorDataset]]:
        """
        Load and preprocess the nursery dataset.
        
        Returns:
            tuple: (x_train, y_train), (x_test, y_test), feature_names, social_indices, (train_dataset, test_dataset)
        """
        self._download_if_needed()
        
        column_names = ['parents', 'has_nurs', 'form', 'children', 'housing', 
                        'finance', 'social', 'health', 'class']
        
        data = pd.read_csv(self.data_path, header=None, names=column_names)
        
        social_values = None
        if self.transform_social:
            social_mapping = {'problematic': 1, 'slightly_prob': 0, 'nonprob': 0}
            data['social'] = data['social'].map(social_mapping)
            social_values = [0, 1]
        
        y = data['class']
        X = data.drop('class', axis=1)
        
        social_feature = X['social']
        
        categorical_features = X.columns
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        
        x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state
        )
        
        self.feature_names = encoder.get_feature_names_out(categorical_features)
        
        self.social_indices = [i for i, name in enumerate(self.feature_names) 
                              if name.startswith('social_')]
        
        x_train_tensor = torch.FloatTensor(x_train_np)
        y_train_tensor = torch.LongTensor(pd.get_dummies(y_train_np).values)
        x_test_tensor = torch.FloatTensor(x_test_np)
        y_test_tensor = torch.LongTensor(pd.get_dummies(y_test_np).values)
        
        self.train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        self.test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        
        self.x_train = x_train_np
        self.y_train = y_train_np
        self.x_test = x_test_np
        self.y_test = y_test_np
        
        return ((self.x_train, self.y_train), 
                (self.x_test, self.y_test), 
                self.feature_names, 
                self.social_indices, 
                (self.train_dataset, self.test_dataset))

    def prepare_attack_data(self, attack_feature: int, train_ratio: float = 0.5) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray,
            np.ndarray, np.ndarray, np.ndarray,
            np.ndarray]:
        """
        Prepare data for attribute inference attacks.
        
        Args:
            attack_feature: Index of the feature to attack
            train_ratio: Proportion of training data to use for attack training
            
        Returns:
            tuple: Attack training and testing data
        """
        if self.x_train is None:
            raise ValueError("Data hasn't been loaded yet. Call load() first.")
            
        attack_train_size = int(len(self.x_train) * train_ratio)
        
        attack_x_train = self.x_train[:attack_train_size]
        
        attack_x_train_feature = attack_x_train[:, attack_feature]
        
        attack_x_train_without_feature = np.delete(attack_x_train, attack_feature, axis=1)
        
        attack_x_test = self.x_train[attack_train_size:]
        attack_x_test_feature = attack_x_test[:, attack_feature]
        attack_x_test_without_feature = np.delete(attack_x_test, attack_feature, axis=1)
        
        unique_values = np.unique(attack_x_train_feature)
        
        return (attack_x_train, attack_x_train_feature, attack_x_train_without_feature,
                attack_x_test, attack_x_test_feature, attack_x_test_without_feature,
                unique_values)

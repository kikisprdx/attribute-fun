"""
Utility functions to fix target tensors for PyTorch loss functions.
"""
from typing import Dict, Tuple, Any, Union

import torch
import numpy as np


def fix_target_tensor(target_feature: np.ndarray) -> Tuple[np.ndarray, Dict[Any, int]]:
    """
    Fix a target tensor to be compatible with CrossEntropyLoss.
    Maps values to continuous integers starting from 0.
    
    Args:
        target_feature: Target feature values (numpy array)
        
    Returns:
        tuple: (target_indices, value_map) where target_indices is a numpy array
              with values remapped to [0, n_classes-1] and value_map is the 
              mapping from original values to indices
    """
    unique_values = np.unique(target_feature)
    
    value_map = {val: idx for idx, val in enumerate(unique_values)}
    
    target_indices = np.array([value_map[val] for val in target_feature])
    
    print(f"Original values: {unique_values}")
    print(f"Mapped to indices: {list(value_map.values())}")
    print(f"Target indices range: [{target_indices.min()}, {target_indices.max()}]")
    
    return target_indices, value_map

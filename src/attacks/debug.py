"""
Quick debugging script to diagnose the target indices issue.
"""
from typing import Tuple

import torch
import numpy as np


def debug_target_indices(target_feature: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Debug the target feature indices issue.
    
    Args:
        target_feature: The target feature values
        
    Returns:
        tuple: Original values, mapped indices, unique values
    """
    print(f"Target feature shape: {target_feature.shape}")
    print(f"Target feature dtype: {target_feature.dtype}")
    print(f"Target feature min: {target_feature.min()}")
    print(f"Target feature max: {target_feature.max()}")
    
    unique_values = np.unique(target_feature)
    print(f"Unique values: {unique_values}")
    
    if (target_feature < 0).any():
        print("WARNING: Target feature contains negative values!")
        print(f"Negative values count: {np.sum(target_feature < 0)}")
    
    value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
    print(f"Value to index mapping: {value_to_idx}")
    
    target_indices = np.array([value_to_idx[val] for val in target_feature])
    print(f"Indices shape: {target_indices.shape}")
    print(f"Indices min: {target_indices.min()}")
    print(f"Indices max: {target_indices.max()}")
    
    return target_feature, target_indices, unique_values


def prepare_for_cross_entropy(target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure target tensor is suitable for CrossEntropyLoss.
    
    Args:
        target_tensor: Target tensor
        
    Returns:
        torch.Tensor: Fixed target tensor
    """
    if target_tensor.min() < 0:
        print("Fixing negative values in target tensor")
        target_tensor = target_tensor - target_tensor.min()
    
    unique_values = torch.unique(target_tensor)
    if len(unique_values) != unique_values.max() + 1:
        print("Remapping values to be contiguous from 0")
        value_map = {val.item(): idx for idx, val in enumerate(unique_values)}
        new_tensor = torch.tensor([value_map[val.item()] for val in target_tensor], 
                                  dtype=target_tensor.dtype)
        return new_tensor
    
    return target_tensor

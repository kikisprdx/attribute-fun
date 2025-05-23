"""
Evaluation utilities for attribute inference attacks.
"""
from typing import Dict, Tuple, Union, List, Any

import numpy as np


def calculate_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        predicted: Predicted values
        actual: Actual values
        
    Returns:
        Accuracy score
    """
    return np.mean(predicted == actual)


def calculate_precision_recall(predicted: np.ndarray, actual: np.ndarray, 
                               positive_value: int = 1) -> Tuple[float, float]:
    """
    Calculate precision and recall for binary classification.
    
    Args:
        predicted: Predicted values
        actual: Actual values
        positive_value: Value to consider as positive class
        
    Returns:
        Tuple containing (precision, recall)
    """
    predicted = np.array(predicted)
    actual = np.array(actual)
    
    tp = np.sum((predicted == positive_value) & (actual == positive_value))
    fp = np.sum((predicted == positive_value) & (actual != positive_value))
    fn = np.sum((predicted != positive_value) & (actual == positive_value))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall


def calculate_mse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate Mean Squared Error for regression.
    
    Args:
        predicted: Predicted values
        actual: Actual values
        
    Returns:
        MSE score
    """
    return np.mean((predicted - actual) ** 2)


def evaluate_attack(predicted: np.ndarray, actual: np.ndarray, 
                   is_categorical: bool = True, positive_value: int = 1) -> Dict[str, float]:
    """
    Evaluate an attribute inference attack with appropriate metrics.
    
    Args:
        predicted: Predicted values
        actual: Actual values
        is_categorical: Whether the attack feature is categorical
        positive_value: Value to consider as positive class for binary metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    results: Dict[str, float] = {}
    
    results['accuracy'] = calculate_accuracy(predicted, actual)
    
    if is_categorical:
        if len(np.unique(actual)) == 2:
            precision, recall = calculate_precision_recall(predicted, actual, positive_value)
            results['precision'] = precision
            results['recall'] = recall
            
            if precision + recall > 0:
                results['f1_score'] = 2 * precision * recall / (precision + recall)
            else:
                results['f1_score'] = 0
    else:
        results['mse'] = calculate_mse(predicted, actual)
        
    return results

"""
White-box attribute inference attack implementation.
"""
from typing import List, Optional, Tuple, Union, Any, Sequence

import torch
import numpy as np


class AttributeInferenceWhiteBox:
    """
    White-box attribute inference attack.

    This attack assumes complete access to the target model's
    parameters and architecture, allowing for gradient-based inference.
    """
    
    def __init__(self, target_model: Any, attack_feature: int) -> None:
        """
        Initialize the white-box attack.

        Args:
            target_model: The target model to attack
            attack_feature: Index of the feature to be attacked
        """
        self.target_model = target_model
        self.attack_feature = attack_feature
        self.is_continuous: bool = False
        
    def infer(self, x: np.ndarray, pred: Optional[np.ndarray] = None, 
              values: Optional[np.ndarray] = None, priors: Optional[List[float]] = None, 
              learning_rate: float = 0.01, iterations: int = 100) -> np.ndarray:
        """
        Infer the attack feature values.

        Args:
            x: Data without the attack feature
            pred: Predictions from the target model (optional)
            values: Possible values of the attack feature
            priors: Prior probabilities of the feature values
            learning_rate: Learning rate for gradient optimization
            iterations: Number of iterations for optimization
            
        Returns:
            Inferred values of the attack feature
        """
        if values is None:
            raise ValueError("Values of the attacked feature must be provided")
        
        if len(values) > 5 or np.issubdtype(values.dtype, np.floating):
            self.is_continuous = True
            print("Detected continuous values for white-box attack")
        else:
            print(f"Detected discrete values: {values}")
        
        if priors is None:
            priors = [1/len(values)] * len(values)
        
        result = np.zeros(len(x))
            
        if self.is_continuous:
            min_val, max_val = np.min(values), np.max(values)
            test_values = np.linspace(min_val, max_val, min(10, len(values)))
            
            for i in range(len(x)):
                best_prob = -1
                best_value = test_values[0]
                
                for val in test_values:
                    sample = x[i].copy()
                    full_sample = np.insert(sample, self.attack_feature, val)
                    
                    try:
                        with torch.no_grad():
                            full_sample_tensor = torch.FloatTensor([full_sample])
                            pred_proba = self.target_model(full_sample_tensor).numpy()[0]
                        
                        prob = np.max(pred_proba)
                        
                        if prob > best_prob:
                            best_prob = prob
                            best_value = val
                    except Exception as e:
                        print(f"Prediction failed for value {val}: {e}")
                        continue
                
                result[i] = best_value
        else:
            for i in range(len(x)):
                best_prob = -1
                best_value = values[0]
                
                for val_idx, val in enumerate(values):
                    sample = x[i].copy()
                    full_sample = np.insert(sample, self.attack_feature, val)
                    
                    try:
                        with torch.no_grad():
                            full_sample_tensor = torch.FloatTensor([full_sample])
                            pred_proba = self.target_model(full_sample_tensor).numpy()[0]
                        
                        prob = np.max(pred_proba) * priors[val_idx]
                        
                        if prob > best_prob:
                            best_prob = prob
                            best_value = val
                    except Exception as e:
                        print(f"Prediction failed for value {val}: {e}")
                        if val_idx < len(priors) and priors[val_idx] > best_prob:
                            best_prob = priors[val_idx]
                            best_value = val
                
                result[i] = best_value
        
        return result.reshape(1, -1)

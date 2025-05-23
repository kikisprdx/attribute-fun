"""
Visualization utilities for attribute inference attacks.
"""
from typing import List, Optional, Union, Any, Sequence

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_feature_importance(model: Any, feature_names: np.ndarray, attack_feature_idx: int) -> None:
    """
    Plot feature importance for the attack model.
    
    Args:
        model: Trained attack model
        feature_names: Names of all features
        attack_feature_idx: Index of the attack feature
    """
    remaining_features = np.delete(feature_names, attack_feature_idx)
    
    if hasattr(model, 'layer1'):
        weights = model.layer1.weight.data.numpy()
        
        importance = np.mean(np.abs(weights), axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': list(remaining_features) + ['Model Predictions'],
            'Importance': importance
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Feature Importance for Attribute Inference Attack')
        plt.tight_layout()
        plt.show()
    else:
        print("Model doesn't have a simple structure for extracting feature importance.")


def visualize_information_leakage(target_model: Any, data: np.ndarray, 
                                 feature_values: np.ndarray, 
                                 attack_feature_idx: int) -> None:
    """
    Visualize how model predictions leak information about the sensitive attribute.
    
    Args:
        target_model: The target model
        data: The data including the sensitive attribute
        feature_values: Possible values of the sensitive attribute
        attack_feature_idx: Index of the sensitive attribute
    """
    sensitive_attr = data[:, attack_feature_idx]
    
    data_without_feature = np.delete(data, attack_feature_idx, axis=1)
    
    predictions = target_model.predict_proba(data_without_feature)
    
    avg_preds_by_sensitive = []
    
    for value in feature_values:
        indices = sensitive_attr == value
        
        avg_pred = np.mean(predictions[indices], axis=0)
        avg_preds_by_sensitive.append(avg_pred)
    
    avg_preds_df = pd.DataFrame(
        avg_preds_by_sensitive,
        index=[f'Sensitive Attr = {v}' for v in feature_values]
    )
    
    plt.figure(figsize=(12, 6))
    avg_preds_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Average Model Predictions by Sensitive Attribute Value')
    plt.xlabel('Sensitive Attribute Value')
    plt.ylabel('Average Prediction Probability')
    plt.legend(title='Prediction Class')
    plt.tight_layout()
    plt.show()


def visualize_attack_confidence(predicted_probs: np.ndarray, actual_values: np.ndarray, 
                               positive_value: int = 1) -> None:
    """
    Visualize attack success by confidence level.
    
    Args:
        predicted_probs: Predicted probabilities from the attack model
        actual_values: Actual values of the sensitive attribute
        positive_value: Value to consider as positive class
    """
    predicted_probs = np.array(predicted_probs)
    actual_values = np.array(actual_values)
    
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    bin_indices = np.digitize(predicted_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    bin_accuracy = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        bin_mask = bin_indices == i
        
        if np.sum(bin_mask) > 0:
            bin_preds = (predicted_probs[bin_mask] >= 0.5).astype(int)
            bin_actual = (actual_values[bin_mask] == positive_value).astype(int)
            
            acc = np.mean(bin_preds == bin_actual)
            bin_accuracy.append(acc)
            bin_counts.append(np.sum(bin_mask))
        else:
            bin_accuracy.append(0)
            bin_counts.append(0)
    
    bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins) - 1)]
    
    plt.figure(figsize=(12, 6))
    
    ax1 = plt.gca()
    ax1.bar(bin_labels, bin_accuracy, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Confidence Range')
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1)
    
    ax2 = ax1.twinx()
    ax2.plot(bin_labels, bin_counts, 'r-', marker='o')
    ax2.set_ylabel('Number of Samples', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Attack Accuracy by Confidence Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(predicted: np.ndarray, actual: np.ndarray, 
                         class_names: Optional[List[str]] = None) -> None:
    """
    Plot confusion matrix for classification results.
    
    Args:
        predicted: Predicted values
        actual: Actual values
        class_names: Names of classes (optional)
    """
    predicted = np.array(predicted)
    actual = np.array(actual)
    
    classes = np.unique(np.concatenate((predicted, actual)))
    n_classes = len(classes)
    
    if class_names is None:
        class_names = [str(c) for c in classes]
    
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i in range(len(actual)):
        true_idx = np.where(classes == actual[i])[0][0]
        pred_idx = np.where(classes == predicted[i])[0][0]
        conf_matrix[true_idx, pred_idx] += 1
    
    conf_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def compare_attack_performance(results_dict):
    """
    Compare performance of different attacks.
    
    Args:
        results_dict: Dictionary mapping attack names to their evaluation metrics
    """
    # Extract metrics for comparison
    attacks = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Create a DataFrame for comparison
    comparison = pd.DataFrame(index=attacks)
    
    for metric in metrics:
        metric_values = []
        for attack in attacks:
            if metric in results_dict[attack]:
                metric_values.append(results_dict[attack][metric])
            else:
                metric_values.append(np.nan)
        
        comparison[metric] = metric_values
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    comparison.plot(kind='bar', figsize=(12, 8))
    plt.title('Attack Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Attack Method')
    plt.ylim(0, 1)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.show()
    
    # Print the table
    print("\nAttack Performance Comparison:")
    print(comparison.round(4))
    
    return comparison

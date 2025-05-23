# %% Cell 1 - Introduction
# # Visualizing How Attribute Inference Attacks Work
#
# This script provides a visual demonstration of how attribute inference attacks work.
# We'll show:
#
# 1. How a target model's predictions leak information about a sensitive attribute
# 2. How an attack model learns to exploit this leakage
# 3. The evolution of both models' probability distributions during training
#
# This is a simplified educational demonstration using a synthetic dataset.

# %% Cell 2 - Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# %% Cell 3 - Generate synthetic data with a sensitive attribute
# We'll create a dataset where some features correlate with a sensitive attribute

def generate_synthetic_data(n_samples=5000, n_features=10, sensitive_influence=0.7):
    """
    Generate synthetic data where the sensitive attribute influences other features.

    Args:
        n_samples: Number of samples
        n_features: Number of non-sensitive features
        sensitive_influence: How strongly the sensitive attribute influences other features

    Returns:
        X: Feature matrix including the sensitive attribute
        y: Target variable
    """
    # Generate sensitive attribute (binary)
    sensitive_attr = np.random.randint(0, 2, size=n_samples)

    # Generate other features with some correlation to sensitive attribute
    X = np.random.randn(n_samples, n_features)

    # Make some features correlate with sensitive attribute
    for i in range(3):  # First 3 features will be influenced
        X[:, i] = X[:, i] + sensitive_influence * sensitive_attr

    # Generate target variable (3 classes)
    # Class is influenced by features AND sensitive attribute
    logits = 0.5 * X[:, 0] - 0.7 * X[:, 1] + 0.3 * X[:, 2] + 0.8 * sensitive_attr

    # Convert to probabilities and then to classes
    probs = 1 / (1 + np.exp(-logits))
    y = np.zeros(n_samples)
    y[probs < 0.3] = 0
    y[(probs >= 0.3) & (probs < 0.7)] = 1
    y[probs >= 0.7] = 2

    # Insert sensitive attribute as a feature
    X_with_sensitive = np.insert(X, 0, sensitive_attr, axis=1)

    return X_with_sensitive, y.astype(int)

# Generate data
print("Generating synthetic data...")
X, y = generate_synthetic_data(n_samples=5000)
print(f"Data shape: {X.shape}, Target shape: {y.shape}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define sensitive attribute index
sensitive_idx = 0  # First column

# Create PyTorch datasets
def create_torch_datasets(X_train, X_test, y_train, y_test):
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset, test_dataset

train_dataset, test_dataset = create_torch_datasets(X_train, X_test, y_train, y_test)

# %% Cell 4 - Define the target model
class TargetModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super(TargetModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x

# %% Cell 5 - Define the attack model
class AttackModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(AttackModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 2)  # Binary classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x

# %% Cell 6 - Train the target model
def train_target_model(model, train_dataset, num_epochs=10, batch_size=64, learning_rate=0.001):
    # Create data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    epoch_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)

        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return model, epoch_losses

# Initialize and train the target model
print("\nTraining target model...")
target_model = TargetModel(input_size=X_train.shape[1])
target_model, target_losses = train_target_model(target_model, train_dataset, num_epochs=10)

# %% Cell 7 - Evaluate the target model's accuracy
def evaluate_model(model, test_dataset, batch_size=64):
    # Create data loader
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluation
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    return accuracy, all_preds, all_labels

# Evaluate the target model
target_accuracy, _, _ = evaluate_model(target_model, test_dataset)
print(f"Target model accuracy: {target_accuracy:.4f}")

# %% Cell 8 - Analyze how target model leaks information about sensitive attribute
def analyze_information_leakage(model, data, sensitive_idx):
    """
    Analyze how a model's predictions differ based on sensitive attribute values.
    """
    # Extract sensitive attribute
    sensitive_attr = data[:, sensitive_idx]

    # Create a copy of data for prediction
    # Instead of removing the sensitive attribute, we'll just get it for analysis
    # but keep the full feature set for the model input

    # Get model predictions
    with torch.no_grad():
        inputs = torch.FloatTensor(data)
        outputs = model(inputs).numpy()

    # Calculate average predictions for each sensitive attribute value
    unique_values = np.unique(sensitive_attr)
    avg_preds = {}

    for value in unique_values:
        mask = sensitive_attr == value
        avg_preds[f"Sensitive={int(value)}"] = np.mean(outputs[mask], axis=0)

    # Create bar chart
    plt.figure(figsize=(10, 6))

    # Create positions for grouped bars
    n_groups = avg_preds[f"Sensitive={int(unique_values[0])}"].shape[0]
    positions = np.arange(n_groups)
    width = 0.35

    # Plot bars for each sensitive attribute value
    for i, value in enumerate(unique_values):
        key = f"Sensitive={int(value)}"
        plt.bar(positions + i*width, avg_preds[key], width,
                label=key, alpha=0.7)

    plt.xlabel('Target Class')
    plt.ylabel('Average Prediction Probability')
    plt.title('Target Model Prediction Distribution by Sensitive Attribute')
    plt.xticks(positions + width/2, [f'Class {i}' for i in range(n_groups)])
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate difference in distributions
    dist_diff = np.sum(np.abs(avg_preds[f"Sensitive=0"] - avg_preds[f"Sensitive=1"]))
    print(f"Distribution difference: {dist_diff:.4f}")

    return avg_preds

# Analyze how target model leaks information about sensitive attribute
target_prediction_dists = analyze_information_leakage(target_model, X_test, sensitive_idx)

# %% Cell 9 - Prepare data for the attack model
def prepare_attack_data(target_model, data, sensitive_idx):
    """
    Prepare data for attack model by combining non-sensitive features with target model predictions.
    """
    # Extract sensitive attribute (the target for the attack)
    sensitive_attr = data[:, sensitive_idx]

    # Get target model predictions (using full data)
    with torch.no_grad():
        inputs = torch.FloatTensor(data)
        outputs = target_model(inputs)
        predictions = outputs.numpy()

    # Remove sensitive attribute from input features for attack input
    data_without_sensitive = np.delete(data, sensitive_idx, axis=1)

    # Combine remaining features with target model predictions
    attack_inputs = np.concatenate((data_without_sensitive, predictions), axis=1)

    return attack_inputs, sensitive_attr

# Prepare attack data
print("\nPreparing data for attack model...")
attack_X_train, attack_y_train = prepare_attack_data(target_model, X_train, sensitive_idx)
attack_X_test, attack_y_test = prepare_attack_data(target_model, X_test, sensitive_idx)

# Create PyTorch datasets for attack model
attack_train_dataset = TensorDataset(
    torch.FloatTensor(attack_X_train),
    torch.LongTensor(attack_y_train)
)
attack_test_dataset = TensorDataset(
    torch.FloatTensor(attack_X_test),
    torch.LongTensor(attack_y_test)
)

# %% Cell 10 - Train attack model and visualize the learning process
def train_attack_model_with_visualization(model, train_dataset, test_dataset,
                                         target_model, X_test, sensitive_idx,
                                         num_epochs=20, batch_size=64, learning_rate=0.001,
                                         visualization_epochs=5, save_interval=2):
    """
    Train the attack model and visualize its learning process periodically.
    """
    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # For visualization
    attack_prediction_dists_over_time = []
    attack_accuracies = []
    attack_losses = []
    saved_models = []
    saved_epochs = []

    visualize_at_epochs = [0] + [i for i in range(1, num_epochs+1, num_epochs//visualization_epochs)]

    # Create an untrained clone of the model for visualization
    # Extract the actual hidden size from the original model
    hidden_size = model.layer1.out_features  # This gets the actual hidden size used in the original model
    untrained_model = AttackModel(input_size=model.layer1.in_features, hidden_size=hidden_size)
    untrained_model.load_state_dict(model.state_dict())
    saved_models.append(untrained_model)
    saved_epochs.append(0)

    # Initial distribution (untrained model)
    initial_attack_dist = analyze_attack_distribution(untrained_model, target_model, X_test, sensitive_idx, show_plot=False)
    attack_prediction_dists_over_time.append(initial_attack_dist)

    # Initial accuracy
    initial_acc, _, _ = evaluate_model(model, test_dataset)
    attack_accuracies.append(initial_acc)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        attack_losses.append(epoch_loss)

        # Evaluate
        epoch_acc, _, _ = evaluate_model(model, test_dataset)
        attack_accuracies.append(epoch_acc)

        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # Save model at regular intervals for ridge plot
        if (epoch+1) % save_interval == 0 or (epoch+1) == num_epochs:
            # Create a clone of the current model using the same architecture
            hidden_size = model.layer1.out_features  # Get actual hidden size from original model
            saved_model = AttackModel(input_size=model.layer1.in_features, hidden_size=hidden_size)
            saved_model.load_state_dict(model.state_dict())
            saved_models.append(saved_model)
            saved_epochs.append(epoch+1)

            # For backwards compatibility with existing code
            if (epoch+1) in visualize_at_epochs:
                epoch_dist = analyze_attack_distribution(
                    model, target_model, X_test, sensitive_idx,
                    show_plot=False
                )
                attack_prediction_dists_over_time.append(epoch_dist)

    # Create visualization of learning progress
    visualize_attack_learning(
        target_prediction_dists,
        attack_prediction_dists_over_time,
        visualize_at_epochs,
        attack_accuracies
    )

    return model, attack_accuracies, attack_losses, saved_models, saved_epochs

def analyze_attack_distribution(attack_model, target_model, data, sensitive_idx, show_plot=True):
    """
    Analyze how the attack model's predictions vary by true sensitive attribute values.
    """
    # Prepare data for attack model
    attack_X, attack_y = prepare_attack_data(target_model, data, sensitive_idx)

    # Get attack model predictions
    with torch.no_grad():
        inputs = torch.FloatTensor(attack_X)
        outputs = attack_model(inputs)
        outputs = outputs.numpy()

    # Calculate average predictions for different sensitive values
    unique_values = np.unique(attack_y)
    avg_preds = {}

    for value in unique_values:
        mask = attack_y == value
        avg_preds[f"True={int(value)}"] = np.mean(outputs[mask], axis=0)

    if show_plot:
        # Create bar chart
        plt.figure(figsize=(10, 6))

        # Create positions for grouped bars
        n_groups = avg_preds[f"True={int(unique_values[0])}"].shape[0]
        positions = np.arange(n_groups)
        width = 0.35

        # Plot bars for each true sensitive value
        for i, value in enumerate(unique_values):
            key = f"True={int(value)}"
            plt.bar(positions + i*width, avg_preds[key], width,
                    label=key, alpha=0.7)

        plt.xlabel('Predicted Class (Sensitive Attribute)')
        plt.ylabel('Average Prediction Probability')
        plt.title('Attack Model Prediction Distribution by True Sensitive Value')
        plt.xticks(positions + width/2, [f'Sensitive={i}' for i in range(n_groups)])
        plt.legend()
        plt.tight_layout()
        plt.show()

    return avg_preds

def visualize_attack_learning(target_dists, attack_dists_over_time, epochs, accuracies):
    """
    Visualize how the attack model learns to exploit information leakage over time.
    """
    # Number of visualization points
    n_points = len(attack_dists_over_time)

    # Create a figure with subplots
    fig, axes = plt.subplots(n_points, 2, figsize=(15, 5*n_points))

    for i in range(n_points):
        # Plot target model distribution in the left column
        ax1 = axes[i, 0]

        # Target model bars
        n_classes = len(target_dists["Sensitive=0"])
        positions = np.arange(n_classes)
        width = 0.35

        ax1.bar(positions, target_dists["Sensitive=0"], width, label="Sensitive=0", alpha=0.7)
        ax1.bar(positions+width, target_dists["Sensitive=1"], width, label="Sensitive=1", alpha=0.7)

        ax1.set_xlabel('Target Class')
        ax1.set_ylabel('Average Prediction Probability')
        ax1.set_title(f'Target Model Distribution')
        ax1.set_xticks(positions + width/2)
        ax1.set_xticklabels([f'Class {j}' for j in range(n_classes)])
        ax1.legend()

        # Plot attack model distribution in the right column
        ax2 = axes[i, 1]

        # Attack model bars
        attack_dist = attack_dists_over_time[i]
        n_sens_classes = len(attack_dist["True=0"])
        positions2 = np.arange(n_sens_classes)

        ax2.bar(positions2, attack_dist["True=0"], width, label="True=0", alpha=0.7)
        ax2.bar(positions2+width, attack_dist["True=1"], width, label="True=1", alpha=0.7)

        epoch_label = "Untrained" if i == 0 else f"Epoch {epochs[i]}"
        ax2.set_xlabel('Predicted Sensitive Value')
        ax2.set_ylabel('Average Prediction Probability')
        ax2.set_title(f'Attack Model Distribution ({epoch_label}) - Accuracy: {accuracies[i]:.4f}')
        ax2.set_xticks(positions2 + width/2)
        ax2.set_xticklabels([f'Sensitive={j}' for j in range(n_sens_classes)])
        ax2.legend()

    plt.tight_layout()
    plt.show()

# Initialize the attack model
attack_model = AttackModel(input_size=attack_X_train.shape[1])

# Train attack model with visualization
print("\nTraining attack model with visualization...")
attack_model, attack_accuracies, attack_losses, saved_models, saved_epochs = train_attack_model_with_visualization(
    attack_model,
    attack_train_dataset,
    attack_test_dataset,
    target_model,
    X_test,
    sensitive_idx,
    num_epochs=20,
    visualization_epochs=4,
    save_interval=2  # Save model every 2 epochs
)

# %% Cell 11 - Evaluate final attack performance
def evaluate_attack_performance(model, test_dataset):
    """
    Evaluate attack model performance in detail.
    """
    # Create data loader
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # Collect predictions and labels
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probs, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')

    print("\nFinal Attack Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = pd.crosstab(
        pd.Series(all_labels, name='Actual'),
        pd.Series(all_preds, name='Predicted')
    )
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Attack Model Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Plot ROC curve (using prediction probabilities)
    plt.figure(figsize=(8, 6))
    thresholds = np.linspace(0, 1, 100)
    tpr_values = []
    fpr_values = []

    binary_labels = np.array(all_labels)
    binary_probs = np.array(all_probs)

    for threshold in thresholds:
        binary_preds = (binary_probs >= threshold).astype(int)
        tp = np.sum((binary_preds == 1) & (binary_labels == 1))
        fp = np.sum((binary_preds == 1) & (binary_labels == 0))
        tn = np.sum((binary_preds == 0) & (binary_labels == 0))
        fn = np.sum((binary_preds == 0) & (binary_labels == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_values.append(tpr)
        fpr_values.append(fpr)

    plt.plot(fpr_values, tpr_values, marker='.')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Attack Model')
    plt.grid(True)
    plt.show()

# Evaluate final attack performance
evaluate_attack_performance(attack_model, attack_test_dataset)

# %% Cell 12 - Ridge plot visualization
def create_ridge_plot(target_model, attack_models, data, sensitive_idx, epochs):
    """
    Create a ridge plot showing probability distributions for target and attack models at different training stages.
    """
    plt.figure(figsize=(14, 3 + 1.5 * len(attack_models)))

    # Prepare target model predictions by sensitive attribute value
    sensitive_attr = data[:, sensitive_idx]
    unique_values = np.unique(sensitive_attr)

    # Get target model predictions using complete data
    with torch.no_grad():
        target_outputs = target_model(torch.FloatTensor(data)).numpy()

    # Create dataframe for ridge plot
    ridge_data = []

    # Add target model predictions
    for value in unique_values:
        mask = sensitive_attr == value
        for i, probs in enumerate(target_outputs[mask]):
            for class_idx, prob in enumerate(probs):
                ridge_data.append({
                    'Model': 'Target Model',
                    'Sensitive': f'Value {int(value)}',
                    'Class': f'Class {class_idx}',
                    'Probability': prob
                })

    # Prepare attack data
    attack_X, attack_y = prepare_attack_data(target_model, data, sensitive_idx)

    # Add each attack model's predictions at different epochs
    for model, epoch in zip(attack_models, epochs):
        # Get predictions from this model
        with torch.no_grad():
            outputs = model(torch.FloatTensor(attack_X)).numpy()

        # Add to dataframe
        for value in unique_values:
            mask = attack_y == value
            for j, probs in enumerate(outputs[mask]):
                for class_idx, prob in enumerate(probs):
                    ridge_data.append({
                        'Model': f'Attack Model (Epoch {epoch})',
                        'Sensitive': f'Value {int(value)}',
                        'Class': f'Class {class_idx}',
                        'Probability': prob
                    })

    # Convert to DataFrame
    ridge_df = pd.DataFrame(ridge_data)

    # Sort models in order of increasing epochs
    model_order = ['Target Model'] + [f'Attack Model (Epoch {e})' for e in sorted(epochs)]
    
    # Create a custom color palette with descriptive labels
    palette = sns.color_palette("Set1", n_colors=len(unique_values))
    
    # Use simpler parameters for better compatibility
    grid = sns.FacetGrid(ridge_df, row="Model", hue="Sensitive", aspect=4, height=2.0,
                        palette=palette, row_order=model_order)
    
    # Increase spacing between subplots
    grid.fig.subplots_adjust(hspace=0.5)  # Increase vertical spacing between subplots
    
    # Map the density plot
    grid.map_dataframe(sns.kdeplot, x="Probability", fill=True, alpha=0.7)
    
    # Add vertical lines at 0.5 probability
    grid.map(plt.axvline, x=0.5, linestyle="--", color="k", alpha=0.5)
    
    # Customize the plot
    grid.set_titles("{row_name}", fontsize=12, fontweight='bold')
    grid.set_axis_labels("Prediction Probability", "Density", fontsize=12)
    
    # Add a more descriptive legend with clearer labels - using simpler styling
    handles, _ = plt.gca().get_legend_handles_labels()
    # Create custom labels for each sensitive value
    labels = [f'Sensitive Attribute = {int(v)}' for v in unique_values]
    # Place legend outside the plot
    leg = grid.fig.legend(handles, labels, title="Sensitive Attribute Value", 
                         loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    grid.fig.suptitle('Evolution of Prediction Distributions During Attack Training', y=1.02, fontsize=16, fontweight='bold')

    # Adjust the layout
    plt.tight_layout()
    plt.show()

# Create a ridge plot showing evolution of distributions across epochs
try:
    # This will run if saved_models exists (after training)
    create_ridge_plot(
        target_model,
        saved_models,  # Use all saved models at different training stages
        X_test,
        sensitive_idx,
        saved_epochs  # Pass all the corresponding epoch numbers
    )
except NameError:
    # Fallback if training hasn't completed yet
    print("Ridge plot requires saved models from training. Run training cell first.")

# %% Cell 13 - Conclusion
# # Understanding Attribute Inference Attacks
#
# This demonstration has shown how:
#
# 1. Models implicitly learn correlations with sensitive attributes, even when these attributes are excluded from the input
# 2. These correlations manifest as different prediction distributions depending on the sensitive attribute value
# 3. Attack models can exploit these distribution differences to infer the sensitive attribute
# 4. As the attack model trains, it gets better at distinguishing the patterns in target model predictions
#
# ## Privacy Implications
#
# This demonstrates a fundamental privacy challenge in machine learning: removing sensitive attributes from the data
# is not sufficient protection if other features correlate with those attributes.
#
# More robust privacy protections, such as differential privacy techniques, are needed to prevent
# information leakage through model predictions.

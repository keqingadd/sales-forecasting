import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device, scaler=None):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    if scaler:
        all_preds = scaler.inverse_transform(all_preds.reshape(-1, 1))
        all_targets = scaler.inverse_transform(all_targets.reshape(-1, 1))
    
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    
    mask = all_targets != 0
    mape = np.mean(np.abs((all_targets[mask] - all_preds[mask]) / all_targets[mask])) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': all_preds,
        'targets': all_targets
    }

def visualize_predictions(targets, predictions, model_name):
    """Visualize predictions vs actual values."""
    plt.figure(figsize=(12, 6))
    plt.plot(targets, label='Actual Sales', alpha=0.7)
    plt.plot(predictions, label='Predicted Sales', alpha=0.7)
    plt.title(f'{model_name} - Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/{model_name}_predictions.png')
    plt.show()

def plot_error_distribution(targets, predictions, model_name):
    """Plot prediction error distribution."""
    errors = targets - predictions
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'{model_name} - Prediction Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'results/{model_name}_error_distribution.png')
    plt.show()
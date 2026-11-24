import torch
import os
from src.data_loader import load_alcohol_sales_data, visualize_sales_data
from src.models import get_model
from src.train import train_model, plot_training_history
from src.evaluate import evaluate_model, visualize_predictions, plot_error_distribution
from src.utils import set_device, save_model

def main():
    os.makedirs('results', exist_ok=True)
    
    device = set_device()
    
    print("Loading Alcohol Sales dataset...")
    try:
        train_loader, test_loader, scaler, sales_data = load_alcohol_sales_data(batch_size=32)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download the dataset and try again.")
        return
    
    print("Visualizing sales data...")
    visualize_sales_data(sales_data)
    
    models = {
        'lstm': get_model('lstm', input_size=1, hidden_size=50, num_layers=2),
        'gru': get_model('gru', input_size=1, hidden_size=50, num_layers=2)
    }
    
    num_epochs = 100
    learning_rate = 0.001
    
    results = {}
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        model = model.to(device)
        
        train_losses, test_losses = train_model(
            model, train_loader, test_loader, device,
            num_epochs=num_epochs, learning_rate=learning_rate
        )
        
        plot_training_history(train_losses, test_losses, model_name)
        
        model.load_state_dict(torch.load('results/best_model.pth'))
        metrics = evaluate_model(model, test_loader, device, scaler)
        
        print(f"\n{model_name} Results:")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        
        visualize_predictions(metrics['targets'], metrics['predictions'], model_name)
        
        plot_error_distribution(metrics['targets'], metrics['predictions'], model_name)
        
        results[model_name] = metrics
        
        save_model(model, f'results/{model_name}_final.pth')
    
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    
    for model_name, metrics in results.items():
        print(f"{model_name}: MSE={metrics['mse']:.6f}, RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}")
    
    with open('results/comparison.txt', 'w') as f:
        for model_name, metrics in results.items():
            f.write(f"{model_name}: MSE={metrics['mse']:.6f}, RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}\n")

if __name__ == '__main__':
    main()
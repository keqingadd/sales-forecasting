import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

class SalesDataset(Dataset):
    """Custom dataset for sales time series."""
    def __init__(self, data, seq_length=12):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])

def load_alcohol_sales_data(data_dir='./data', seq_length=12, batch_size=32):
    """Load and preprocess Alcohol Sales dataset."""
    os.makedirs(data_dir, exist_ok=True)
    
    data_path = os.path.join(data_dir, 'Alcohol_Sales.csv')
    
    if not os.path.exists(data_path):
        print("Dataset not found. Please download Alcohol_Sales.csv from:")
        print("https://www.kaggle.com/datasets/bulentsiyah/for-simple-exercises-time-series-forecasting")
        print(f"Place it in {data_path}")
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    sales_column = None
    for col in df.columns:
        if 'sale' in col.lower():
            sales_column = col
            break
    
    if sales_column is None:
        sales_column = df.columns[1]
    
    sales_data = df[sales_column].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    sales_normalized = scaler.fit_transform(sales_data)
    
    train_size = int(len(sales_normalized) * 0.9)
    train_data = sales_normalized[:train_size]
    test_data = sales_normalized[train_size:]
    
    train_dataset = SalesDataset(train_data, seq_length)
    test_dataset = SalesDataset(test_data, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler, sales_data

def visualize_sales_data(sales_data):
    """Visualize sales data as line chart."""
    plt.figure(figsize=(12, 6))
    plt.plot(sales_data)
    plt.title('Alcohol Sales Over Time')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.savefig('results/sales_data.png')
    plt.show()
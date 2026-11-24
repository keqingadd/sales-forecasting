# Sales Forecasting

Implementation of two RNN variants (LSTM and GRU) for sales volume prediction on Alcohol Sales dataset.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn
- tqdm

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Download Alcohol_Sales.csv from:
https://www.kaggle.com/datasets/bulentsiyah/for-simple-exercises-time-series-forecasting

Place it in `data/Alcohol_Sales.csv`

## Usage

1. Run the main script:
```bash
python main.py
```

2. The script will:
   - Load and preprocess Alcohol Sales dataset
   - Train both LSTM and GRU models
   - Evaluate models on test set
   - Generate visualizations and comparison results

## Results

- **LSTM**: Expected MSE < 0.02
- **GRU**: Expected MSE < 0.02

Both models should achieve MSE < 0.02 as required.

## File Structure

- `src/data_loader.py`: Data loading and preprocessing
- `src/models.py`: Model definitions (LSTM, GRU)
- `src/train.py`: Training pipeline
- `src/evaluate.py`: Evaluation metrics
- `src/utils.py`: Utility functions
- `main.py`: Main execution script
- `results/`: Saved models, plots, and metrics
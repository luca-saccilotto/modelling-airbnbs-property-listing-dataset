# Import libraries
import numpy as np
import pandas as pd
import torch

# Import methods defined previously
from modelling_regression import load_airbnb

# Create a PyTorch dataset that returns a tuple when indexed
class AirbnbNightlyPriceRegressionDataset(torch.utils.data.Dataset):

    """Initialize the dataset and load the input"""
    def __init__(self, data):
        super().__init__()
        self.X, self.y = load_airbnb(data, "Price_Night")
        self.X = self.X.select_dtypes(include = ["int64", "float64"])

    """Retrieve a single data point from the dataset given an index"""
    def __getitem__(self, index):
        features = torch.tensor(self.X.iloc[index])
        label = self.y.iloc[index]
        return (features, label)

    """Return the total number of data points in the dataset"""
    def __len__(self):
        return len(self.X)

data = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv", index_col = "ID")
df = AirbnbNightlyPriceRegressionDataset(data)

# Split the data into training, validation, and test sets
train_data, test_data = torch.utils.data.random_split(df, lengths = [0.8, 0.2])
train_data, validation_data = torch.utils.data.random_split(train_data, lengths = [0.7, 0.3])

# Creating a data loader for the training and testing set that shuffles the data
train_data = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
test_data = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)
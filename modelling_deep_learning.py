# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Import methods defined previously
from tabular_data import load_airbnb

# Create a PyTorch dataset that returns a tuple when indexed
class AirbnbNightlyPriceRegressionDataset(Dataset):

    """Initialize the dataset and load the input"""
    def __init__(self, data):
        super().__init__()
        self.X, self.y = load_airbnb(data, "Price_Night")
        self.X = self.X.select_dtypes(include = ["int64", "float64"])

    """Retrieve a single data point from the dataset given an index"""
    def __getitem__(self, index):
        features = torch.tensor(self.X.iloc[index]).float()
        label = torch.tensor(self.y.iloc[index]).float()
        return (features, label)

    """Return the total number of data points in the dataset"""
    def __len__(self):
        return len(self.X)

data = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv", index_col = "ID")
dataset = AirbnbNightlyPriceRegressionDataset(data)

# Split the data into training, validation, and test sets
train_data, test_data = torch.utils.data.random_split(dataset, lengths = [0.8, 0.2])
train_data, validation_data = torch.utils.data.random_split(train_data, lengths = [0.7, 0.3])

# Creating a data loader for the training, validation and testing set that shuffles the data
train_loader = DataLoader(train_data, batch_size = 12, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 12, shuffle = True)
validation_loader = DataLoader(validation_data, batch_size = 12, shuffle = True)

dataloader = {
    "Train": train_loader,
    "Validation": test_loader,
    "Testing": validation_loader
}

# Create "Neural Network" model in Pytorch
class NeuralNetwork(nn.Module):

    """Initialize the size of input and hidden layers of the network"""
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layers = torch.nn.Sequential(
            nn.Linear(input_size, num_classes),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    """Define the forward pass of the network"""
    def forward(self, input):
        return self.layers(input)

# Train a PyTorch model using the specified dataset and parameters
def train(model, dataloader, epochs = 10):

    """Define the optimizer and TensorBoard writer"""
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    writer = SummaryWriter()

    """Initialize the batch index"""
    batch_idx = 0
    batch_idx2 = 0

    """Loop through the specified number of epochs"""
    for epoch in range(epochs):
        
        """Loop through each batch in the DataLoader"""
        for batch in dataloader["Train"]:
            ## Extract the features and labels from the batch
            features, labels = batch
            ## Compute the model's prediction and the corresponding loss
            prediction = model(features).squeeze()
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            print(round(loss.item(), 3))
            ## Perform an optimization step and zero the gradients
            optimizer.step()
            optimizer.zero_grad()
            ## Visualise the loss function using TensorBoard
            writer.add_scalar("Loss", loss.item(), batch_idx)
            ## Increment the batch index
            batch_idx += 1

        """Loop through each batch in the DataLoader"""
        for batch in dataloader["Validation"]:
            ## Extract the features and labels from the batch
            features, labels = batch
            ## Compute the model's prediction and the corresponding loss
            prediction = model(features).squeeze()
            loss = F.mse_loss(prediction, labels)
            print(round(loss.item(), 3))
            ## Visualise the loss function using TensorBoard
            writer.add_scalar("Loss", loss.item(), batch_idx2)
            ## Increment the batch index
            batch_idx2 += 1

# Ensure that the code inside it is only executed if the script is being run directly
if __name__ == "__main__":
    model = NeuralNetwork(input_size = 11, hidden_size = 1, num_classes = 1)
    train(model, dataloader)
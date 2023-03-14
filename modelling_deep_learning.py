# Import libraries
import yaml
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
    def __init__(self):
        super().__init__()
        data = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv", index_col = "ID")
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

# Create a function that returns a dictionary containing data loaders for each sets
def get_data_loaders(dataset, batch_size):

    """Split the data into training, validation, and test sets"""
    train_data, test_data = torch.utils.data.random_split(dataset, lengths = [0.8, 0.2])
    train_data, validation_data = torch.utils.data.random_split(train_data, lengths = [0.7, 0.3])

    """Create data loaders for each set that shuffles the data"""
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)
    validation_loader = DataLoader(validation_data, batch_size = batch_size, shuffle = True)

    """Add the results to a dictionary"""
    dataloader = {
        "Train": train_loader,
        "Validation": validation_loader,
        "Testing": test_loader
    }

    return dataloader

# Define a function which reads the configuration file
def get_nn_config():
    with open("nn_config.yaml", "r") as stream:
        try:
            nn_config = yaml.safe_load(stream)
            return nn_config
        except yaml.YAMLError as e:
            print(e)

# Create "Neural Network" model in Pytorch
class NeuralNetwork(nn.Module):

    """Initialize the hyperparameters of the network"""
    def __init__(self, config, in_features, out_features):
        super().__init__()
        ## Extract hyperparameters from the configuration file
        hidden_layer_depth = config["hidden_layer_depth"]
        hidden_layer_width = config["hidden_layer_width"]
        ## Create a list to store the layers of the network
        layers = []
        ## Loop over the specified number of hidden layers
        for i in range(hidden_layer_depth):
            ## Create a linear layer with the specified input and output dimensions
            ## The input dimension is the size of the input layer if this is the first hidden layer
            ## Otherwise, it is the width of the previous hidden layer
            ## The output dimension is the width of the hidden layer
            layers.append(nn.Linear(in_features if i == 0 else hidden_layer_width, hidden_layer_width))
            ## Add a ReLU activation function after each hidden layer
            layers.append(nn.ReLU())
        ## Add the output layer with a linear activation function to the list of layers
        layers.append(nn.Linear(hidden_layer_width, out_features))
        ## Create the network as a sequential module using the list of layers
        self.layers = nn.Sequential(*layers)

    """Define the forward pass of the network"""
    def forward(self, input):
        return self.layers(input)

# Train a PyTorch model using the specified dataset and parameters
def train(model, dataloader, config, epochs = 10):

    """Define the optimiser and TensorBoard writer"""
    optimiser_class = getattr(torch.optim, config["optimiser"])
    optimiser = optimiser_class(model.parameters(), lr = config["learning_rate"])
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
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            print(round(loss.item(), 3))
            ## Perform an optimization step and zero the gradients
            optimiser.step()
            optimiser.zero_grad()
            ## Visualise the loss function using TensorBoard
            writer.add_scalars("Train Loss", {"loss": loss.item()}, batch_idx)
            ## Increment the batch index
            batch_idx += 1

        """Loop through each batch in the DataLoader"""
        for batch in dataloader["Validation"]:
            ## Extract the features and labels from the batch
            features, labels = batch
            ## Compute the model's prediction and the corresponding loss
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            print(round(loss.item(), 3))
            ## Visualise the loss function using TensorBoard
            writer.add_scalars("Validation Loss", {"loss": loss.item()}, batch_idx2)
            ## Increment the batch index
            batch_idx2 += 1

# Ensure that the code inside it is only executed if the script is being run directly
if __name__ == "__main__":
    dataset = AirbnbNightlyPriceRegressionDataset()
    dataloaders = get_data_loaders(dataset, batch_size = 12)
    config = get_nn_config()
    model = NeuralNetwork(config, in_features = 11, out_features = 1)
    train(model, dataloaders, config)
# Import libraries
import numpy as np
from tabular_data import load_airbnb
from sklearn import model_selection, linear_model

# Define the path to the file and print its data type
df = "airbnb-property-listings\tabular_data\clean_tabular_data.csv"
print(type(df))

# Load the data into two different variables
X, y = load_airbnb(df, "Price_Night")
X = X.select_dtypes(include = "number")

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size = 0.5)

# Create a simple regression model to predict the nightly cost of each listing
sgdr = linear_model.SGDRegressor()
sgdr.fit(X_train, y_train)

# Compute the R-Squared score of the model on the training data
score = sgdr.score(X_train, y_train)
print("R-Squared:", score)
# Import libraries
import numpy as np
import pandas as pd
from tabular_data import load_airbnb
from sklearn import linear_model, model_selection, metrics

# Define the path to the file that contains the data
df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv", index_col = "ID")

# Load and split the data into training, validation, and test sets
X, y = load_airbnb(df, "Price_Night")
np.random.seed(1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size = 0.5)

print(X_train.shape, y_train.shape)

# Create a simple regression model to predict the nightly cost of each listing
regressor = linear_model.SGDRegressor()
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Calculate the mean squared error of the predictions
mse = metrics.mean_squared_error(y_test, y_pred)
print("MSE: ", mse)

# Calculate the root mean squared error
rmse = np.sqrt(mse)
print("RMSE: ", rmse)

# Compute the R-Squared of the model on the training data
r2 = regressor.score(X_train, y_train)
print("R-Squared: ", r2)
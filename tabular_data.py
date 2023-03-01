# Import libraries
from ast import literal_eval
import numpy as np
import pandas as pd

# Define a function which removes missing values in the rating columns
def remove_rows_with_missing_ratings(df):
    df = df.dropna(subset = ["Cleanliness_rating",
                             "Accuracy_rating",
                             "Communication_rating",
                             "Location_rating",
                             "Check-in_rating",
                             "Value_rating"], how = "any")
    return df

# Define a function that combines the list items into the same string
def combine_description_strings(df: str) -> pd.DataFrame:
    
    def convert_description_strings(x: str) -> str:
        """
        Converts a list of strings to a single string by
        concatenating the remaining strings
        
        """
        try:
            x = literal_eval(x)
            x.remove("About this space")
            x = "".join(x)
            return x
        except:
            print(f"Failed to convert string: {x}")
            return np.nan
    
    df["Description"] = df["Description"].dropna().apply(convert_description_strings)
    df["Description"].replace([r"\\n", "\n", r"\'"], [" "," ",""], regex = True, inplace = True)
    return df

# Define a function to fill missing values with 1 in specific columns
def set_default_feature_values(df):
    cols_to_fill = ["guests", "beds", "bathrooms", "bedrooms"]
    df[cols_to_fill].fillna(value = 1)
    return df

# Encapsulate all the processing code inside a function
def clean_tabular_data(df):
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    return df
"""
Ensure that the code inside it is only executed
if the script is being run as the main program

"""
if __name__ == "__main__":
    path = "airbnb-property-listings/tabular_data/listing.csv"
    raw_data = pd.read_csv(path, index_col = "ID")
    df = clean_tabular_data(raw_data)
    try:
        df.to_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv")
    except:
        print("Failed to write output file")
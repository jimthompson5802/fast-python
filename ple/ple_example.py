import sys

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


# Sample data (2D array with 5 columns)
# data = np.random.randn(10, 2)
data = np.array([[0.25, 0.51], [0.1, 0.2 ], [0.55, 0.3 ], [0.99, 0.49], [.12, .56], [.13, .54]])
print(f"Original data:\n{data}")

df_data = pd.DataFrame(data, columns=[f"f_{i}" for i in range(data.shape[1])])
print(f"Original data_df:\n{df_data}")

# Define the number of bins
num_bins = 4

# Initialize an empty list to store the bin boundaries for each column
bin_boundaries = {}
quantiles = np.linspace(0, 1, num_bins + 1)

# Loop through each column and compute the bin boundaries
for fn in df_data.columns:  # Iterate over columns
    column_data = df_data[fn]  # Extract the current column
    boundaries = column_data.quantile(quantiles) # Compute the bin boundaries for the current column
    bin_boundaries[fn] = boundaries  # Add the bin boundaries to     bin_boundaries.append(boundaries)  # Add the bin boundaries to the list

print(f"\nBin boundaries:")
# Print the bin boundaries for each column
for i, boundaries in bin_boundaries.items():
    print(f"Column {i} bin boundaries:\n{boundaries}")


# Loop through each column and perform piecewise linear encoding
encode_data_list = []
idxs_list = []
encoded_value_list = []
for fn in df_data.columns:  # Iterate over columns
    column_data = df_data[fn].values  # Extract the current column
    column_bin_boundaries = np.array(bin_boundaries[fn])  # Get the bin boundaries for the current column

    # Initialize a matrix of all ones to store the encoded data
    encoded_data = np.ones([column_data.shape[0], num_bins])
    print(f"encoded_data0:\n{encoded_data}")
    
    # Use np.digitize to find the bin indices for each data point
    bin_indices = np.digitize(column_data, column_bin_boundaries) - 1
    print(f"column_bin_boundaries:\n{column_bin_boundaries}")
    print(f"zip(columndata, bin_indices):\n{list(zip(column_data, bin_indices))}")

    # compute numerator, adjust for edge case at max value
    # find the bin min for each data point
    bin_min = column_bin_boundaries[bin_indices]

    # for maximum data point, set bin min to second to last bin boundary
    bin_min[bin_indices == num_bins] = column_bin_boundaries[-2]

    # compute the bin numerator for each data point
    bin_numerator = column_data - bin_min
    print(f"\nbin_numerator:\n{bin_numerator}")
    
    # Calculate the bin widths based on the bin boundaries
    bin_widths = np.diff(column_bin_boundaries)

    # adjust for edge case of last bin
    idxs = bin_indices
    idxs[idxs == num_bins] = num_bins - 1
    # for maximum data point, set bin width to last bin boundary

    bin_denominator = bin_widths[idxs]
    print(f"\nbin_widths:\n{bin_widths}")
   
    # Calculate the encoded value of each data point within the selected bin
    encoded_values = bin_numerator / bin_denominator
   
    # Create a mask to store the encoded value in the corresponding column of encoded_data
    mask = np.zeros_like(encoded_data, dtype=bool)
    mask[np.arange(encoded_data.shape[0]), bin_indices] = True

    # Store the encoded value in the corresponding column of encoded_data
    encoded_data[mask] = encoded_values

    # Create mask to set all values after the column-specific bin index to 0
    mask = np.tile(np.arange(encoded_data.shape[1]), (encoded_data.shape[0], 1))
    mask = mask > bin_indices.reshape(-1, 1)
    encoded_data[mask] = 0


    encode_data_list.append(encoded_data)
    idxs_list.append(bin_indices)
    encoded_value_list.append(encoded_values)

# encoded_data now contains the piecewise linear encoding for each column
encoded_data1 = np.array(encode_data_list)
idxs = np.vstack(idxs_list).T
print(f"Encoded data1: {encoded_data.shape}\n{encoded_data1}")
print(f"idxs: {idxs.shape}\n{idxs}")



class MyTransformer(BaseEstimator, TransformerMixin):
    EPSILON = 1e-8

    def __init__(self, num_bins=4):
        self.num_bins = num_bins    # Store the number of bins
        self.bin_boundaries = {}  # Initialize the bin boundaries

    def fit(self, X, y=None):
        # Fit the transformer to the data
 
        # Initialize an empty list to store the bin boundaries for each column
        quantiles = np.linspace(0, 1, self.num_bins + 1)
        self.feature_names = X.columns

        # Loop through each column and compute the bin boundaries
        for feature_name in self.feature_names:  # Iterate over columns
            column_data = X[feature_name]  # Extract the current column
            boundaries = column_data.quantile(quantiles) # Compute the bin boundaries for the current column
            self.bin_boundaries[feature_name] = boundaries  # Add the bin boundaries to the list

        return self
    
    def transform(self, X):
        # Transform the data using the fitted transformer
        # Loop through each column and perform piecewise linear encoding
        encode_data_list = []
        for feature_name in self.feature_names:  # Iterate over column
            column_data = X[feature_name].values  # Extract the current column
            column_bin_boundaries = np.array(self.bin_boundaries[feature_name])  # Get the bin boundaries for the current column

            # Initialize a matrix of all ones to store the encoded data
            encoded_data = np.ones([column_data.shape[0], self.num_bins])
            
            # Use np.digitize to find the bin indices for each data point
            bin_indices = np.digitize(column_data, column_bin_boundaries) - 1

            # compute numerator, adjust for edge case at max value
            # find the bin min for each data point
            bin_min = column_bin_boundaries[bin_indices]

            # for maximum data point, set bin min to second to last bin boundary
            bin_min[bin_indices == self.num_bins] = column_bin_boundaries[-2]

            # compute the numerator for each data point, x - bin[i-1]
            bin_numerator = column_data - bin_min
            
            # Calculate the bin widths based on the bin boundaries
            bin_widths = np.diff(column_bin_boundaries)

            # adjust for edge case of last bin
            # for maximum data point, set bin width to last bin boundary
            idxs = bin_indices
            idxs[idxs == self.num_bins] = self.num_bins - 1

            # compute the demoninator for each data point: bin[i] - bin[i-1]
            bin_denominator = bin_widths[idxs]
        
            # Calculate the encoded value of each data point within the selected bin
            encoded_values = bin_numerator / bin_denominator
        
            # Create a mask to store the encoded value in the corresponding column of encoded_data
            mask = np.zeros_like(encoded_data, dtype=bool)
            mask[np.arange(encoded_data.shape[0]), bin_indices] = True

            # Store the encoded value in the corresponding column of encoded_data
            encoded_data[mask] = encoded_values

            # Create mask to set all values after the column-specific bin index to 0
            mask = np.tile(np.arange(encoded_data.shape[1]), (encoded_data.shape[0], 1))
            mask = mask > bin_indices.reshape(-1, 1)
            encoded_data[mask] = 0


            encode_data_list.append(encoded_data)

        # stach the encoded data encoded_data into a matrix that contains the piecewise linear encoding for each column
        return np.array(encode_data_list)
    

# Create an instance of the transformer
transformer = MyTransformer(num_bins=4)

# Fit the transformer to the data
transformer.fit(df_data)

# Transform the data using the fitted transformer
encoded_data2 = transformer.transform(df_data)
print(f"sklearn Encoded data: {encoded_data2.shape}\n{encoded_data2}")

print(f"encoded_data1 == encoded_data2: {np.allclose(encoded_data1, encoded_data2)}")

from abc import abstractmethod
import functools
import time
import numpy as np
import pandas as pd
import numba as nb

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_regression

from ple_transformer_cython import _ple_transform_cython


# def jit_wrapper(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         arg_types = tuple(type(arg) for arg in args)
#         kwarg_types = {k: type(v) for k, v in kwargs.items()}
#         print(f"Called with args: {arg_types}, kwargs: {kwarg_types}")
#         # Call the actual JIT-compiled function
#         return func(*args, **kwargs)
#     return wrapper

# _ple_transform = jit_wrapper(_ple_transform)

# force compilation of the function
# _ple_transform(np.array([1., 2., 3.]), np.array([0.25, 0.5, 0.75, 1.0]), 3)

class PiecewiseLinearEncoderBase(BaseEstimator, TransformerMixin):
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
    
    @abstractmethod
    def transform(self, X):
        raise NotImplementedError("transform method requires implementation")
        
class PiecewiseLinearEncoderNumpy(PiecewiseLinearEncoderBase):
    EPSILON = 1e-8
    
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
        return np.array(encode_data_list).astype(np.float32).transpose(1, 0, 2)

class PiecewiseLinearEncoderNumbaV0(PiecewiseLinearEncoderBase):

    @staticmethod
    @nb.njit  #('float32[:,:](float64[:], float64[:], int32)')
    def _ple_transformv0(column_data, column_bin_boundaries, num_bins):
        # print("ple_transform column_data", column_data.shape, "column_bin_boundaries", column_bin_boundaries.shape, "num_bins",num_bins)
        # print("ple_transform column_data", column_data[:5], "column_bin_boundaries", column_bin_boundaries[:5])
        # Initialize a matrix of all ones to store the encoded data
        encoded_data = np.ones((column_data.shape[0], num_bins))
        
        # Use np.digitize to find the bin indices for each data point
        bin_indices = np.digitize(column_data, column_bin_boundaries) - 1

        # compute numerator, adjust for edge case at max value
        # find the bin min for each data point
        bin_min = column_bin_boundaries[bin_indices]

        # for maximum data point, set bin min to second to last bin boundary
        bin_min[bin_indices == num_bins] = column_bin_boundaries[-2]

        # compute the numerator for each data point, x - bin[i-1]
        bin_numerator = column_data - bin_min
        
        # Calculate the bin widths based on the bin boundaries
        bin_widths = np.diff(column_bin_boundaries)

        # adjust for edge case of last bin
        # for maximum data point, set bin width to last bin boundary
        idxs = bin_indices
        idxs[idxs == num_bins] = num_bins - 1

        # compute the demoninator for each data point: bin[i] - bin[i-1]
        bin_denominator = bin_widths[idxs]

        # Calculate the encoded value of each data point within the selected bin
        encoded_values = bin_numerator / bin_denominator

        # Create a mask to store the encoded value in the corresponding column of encoded_data
        mask = np.zeros(encoded_data.shape, dtype=np.bool_)
        # mask[np.arange(encoded_data.shape[0]), bin_indices] = True
        for i in range(encoded_data.shape[0]):
            mask[i, bin_indices[i]] = True

        # Store the encoded value in the corresponding column of encoded_data
        # encoded_data[mask] = encoded_values
        rows, cols = np.where(mask)
        for row, col in zip(rows, cols):
            encoded_data[row, col] = encoded_values[row]
            for i in range(col + 1, encoded_data.shape[1]):
                encoded_data[row, i] = 0

        # # Create mask to set all values after the column-specific bin index to 0
        # mask = np.tile(np.arange(encoded_data.shape[1]), (encoded_data.shape[0], 1))
        # mask = mask > bin_indices.reshape(-1, 1)
        # encoded_data[mask] = 0

        return encoded_data

    def transform(self, X):
        encoded_data_list = []

        # interate over the columns and perform piecewise linear encoding
        for f_n in self.feature_names:
            column_data = X[f_n].values
            column_bin_boundaries = np.array(self.bin_boundaries[f_n])
            # print(f"column_data {column_data.shape} column_bin_boundaries {column_bin_boundaries.shape}")
            # print(f"column_data {type(column_data)} column_bin_boundaries {type(column_bin_boundaries)}")
            encoded_data_list.append(
                self._ple_transformv0(column_data, column_bin_boundaries, self.num_bins)
            )

        return np.array(encoded_data_list).astype(np.float32).transpose(1, 0, 2)

class PiecewiseLinearEncoderNumbaV1(PiecewiseLinearEncoderBase):

    def fit(self, X, y=None):
        super().fit(X, y)

        # force compilation of the function
        self.transform(X[:1])

        return self

    @staticmethod
    @nb.njit  #('float32[:,:](float64[:], float64[:], int32)')
    def _ple_transformv1(column_data, column_bin_boundaries, num_bins):
        # print("ple_transform column_data", column_data.shape, "column_bin_boundaries", column_bin_boundaries.shape, "num_bins",num_bins)
        # print("ple_transform column_data", column_data[:5], "column_bin_boundaries", column_bin_boundaries[:5])
        # Initialize a matrix of all ones to store the encoded data
        encoded_data = np.ones((column_data.shape[0], num_bins))
        
        # Use np.digitize to find the bin indices for each data point
        bin_indices = np.digitize(column_data, column_bin_boundaries) - 1

        # compute numerator, adjust for edge case at max value
        # find the bin min for each data point
        bin_min = column_bin_boundaries[bin_indices]

        # for maximum data point, set bin min to second to last bin boundary
        bin_min[bin_indices == num_bins] = column_bin_boundaries[-2]

        # compute the numerator for each data point, x - bin[i-1]
        bin_numerator = column_data - bin_min
        
        # Calculate the bin widths based on the bin boundaries
        bin_widths = np.diff(column_bin_boundaries)

        # adjust for edge case of last bin
        # for maximum data point, set bin width to last bin boundary
        idxs = bin_indices
        idxs[idxs == num_bins] = num_bins - 1

        # compute the demoninator for each data point: bin[i] - bin[i-1]
        bin_denominator = bin_widths[idxs]

        # Calculate the encoded value of each data point within the selected bin
        encoded_values = bin_numerator / bin_denominator

        # Create a mask to store the encoded value in the corresponding column of encoded_data
        mask = np.zeros(encoded_data.shape, dtype=np.bool_)
        # mask[np.arange(encoded_data.shape[0]), bin_indices] = True
        for i in range(encoded_data.shape[0]):
            mask[i, bin_indices[i]] = True

        # Store the encoded value in the corresponding column of encoded_data
        # encoded_data[mask] = encoded_values
        rows, cols = np.where(mask)
        for row, col in zip(rows, cols):
            encoded_data[row, col] = encoded_values[row]
            for i in range(col + 1, encoded_data.shape[1]):
                encoded_data[row, i] = 0

        # # Create mask to set all values after the column-specific bin index to 0
        # mask = np.tile(np.arange(encoded_data.shape[1]), (encoded_data.shape[0], 1))
        # mask = mask > bin_indices.reshape(-1, 1)
        # encoded_data[mask] = 0

        return encoded_data

    def transform(self, X):
        encoded_data_list = []

        # interate over the columns and perform piecewise linear encoding
        for f_n in self.feature_names:
            column_data = X[f_n].values
            column_bin_boundaries = np.array(self.bin_boundaries[f_n])
            # print(f"column_data {column_data.shape} column_bin_boundaries {column_bin_boundaries.shape}")
            # print(f"column_data {type(column_data)} column_bin_boundaries {type(column_bin_boundaries)}")
            encoded_data_list.append(
                self._ple_transformv1(column_data, column_bin_boundaries, self.num_bins)
            )

        return np.array(encoded_data_list).astype(np.float32).transpose(1, 0, 2)


class PiecewiseLinearEncoderCython(PiecewiseLinearEncoderBase):
    
    def transform(self, X):
        encoded_data_list = []

        # interate over the columns and perform piecewise linear encoding
        for f_n in self.feature_names:
            column_data = X[f_n].values.astype(np.float32)
            column_bin_boundaries = np.array(self.bin_boundaries[f_n]).astype(np.float32)
            # print(f"column_data {column_data.shape} column_bin_boundaries {column_bin_boundaries.shape}")
            # print(f"column_data {type(column_data)} column_bin_boundaries {type(column_bin_boundaries)}")
            # print(f"column_data {column_data.dtype} column_bin_boundaries {column_bin_boundaries}")
            encoded_data_list.append(
                _ple_transform_cython(column_data, column_bin_boundaries, self.num_bins)
            )

        return np.array(encoded_data_list).transpose(1, 0, 2)
    

if __name__ == "__main__":
    NUM_FEATURES = 100
    NUM_BINS = 45
    NUM_SAMPLES = 1_000  #1_000
    # Generate synthetic regression data
    X, y = make_regression(n_samples=NUM_SAMPLES, n_features=NUM_FEATURES, noise=0.1, random_state=1)

    # Convert to pandas DataFrame
    df = pd.DataFrame(data=X, columns=[f'Feature_{i}' for i in range(1, NUM_FEATURES+1)])
    df['Target'] = y

    df_data = df.drop('Target', axis=1).astype(np.float32)

    print(df_data.head())


    # Create an instance of the Numpy transformer
    transformernp = PiecewiseLinearEncoderNP(num_bins=NUM_BINS)

    # Fit the transformer to the data
    transformernp.fit(df_data)

    # Transform the data
    start_time = time.time()
    encoded_data_np = transformernp.transform(df_data)
    end_time = time.time()
    print(f"TransformNP {df_data.shape} took {end_time - start_time} seconds")
    print(f"encoded shape {encoded_data_np.shape} {encoded_data_np.dtype}")

    # Create an instance of the Numba transformer
    transformerv0 = PiecewiseLinearEncoderNumbaV0(num_bins=NUM_BINS)

    # Fit the transformer to the data
    transformerv0.fit(df_data)

    # Transform the data
    start_time = time.time()
    encoded_data_numbav0 = transformerv0.transform(df_data)
    end_time = time.time()
    print(f"\nTransformNumbaV0 {df_data.shape} took {end_time - start_time} seconds")
    print(f"encoded shape {encoded_data_numbav0.shape} {encoded_data_numbav0.dtype}")

    # Create an instance of the Numba transformer
    transformerv1 = PiecewiseLinearEncoderNumbaV1(num_bins=NUM_BINS)

    # Fit the transformer to the data
    transformerv1.fit(df_data)

    # Transform the data
    start_time = time.time()
    encoded_data_numbav1 = transformerv1.transform(df_data)
    end_time = time.time()
    print(f"\nTransformNumbaV1 {df_data.shape} took {end_time - start_time} seconds")
    print(f"encoded shape {encoded_data_numbav1.shape} {encoded_data_numbav1.dtype}")

    # Create an instance of the Cython transformer
    transformercy = PiecewiseLinearEncoderCython(num_bins=NUM_BINS)

    # Fit the transformer to the data
    transformercy.fit(df_data)

    # Transform the data
    start_time = time.time()
    encoded_data_cython = transformercy.transform(df_data)
    end_time = time.time()
    print(f"\nTransformCython {df_data.shape} took {end_time - start_time} seconds")
    print(f"encoded shape {encoded_data_cython.shape} {encoded_data_cython.dtype}")

    print(f"\nnp.allclose(encoded_data_np, encoded_data_numbav0) {np.allclose(encoded_data_np, encoded_data_numbav0)}")
    print(f"\nnp.allclose(encoded_data_np, encoded_data_numbav1) {np.allclose(encoded_data_np, encoded_data_numbav1)}")
    print(f"np.allclose(encoded_data_np, encoded_data_cython) {np.allclose(encoded_data_np, encoded_data_cython)}")

    if not np.allclose(encoded_data_np, encoded_data_cython):
        # print("encoded_data_np != encoded_data_cython")
        # print(f"np.where(encoded_data_np != encoded_data_cython) {np.where(encoded_data_np != encoded_data_cython)}")
        # print(f"encoded_data_np[np.where(encoded_data_np != encoded_data_cython)]\n{encoded_data_np[np.where(encoded_data_np != encoded_data_cython)]}")
        # print(f"encoded_data_cython[np.where(encoded_data_np != encoded_data_cython)]\n{encoded_data_cython[np.where(encoded_data_np != encoded_data_cython)]}")
        # rows, feats, bins = np.where(encoded_data_np != encoded_data_cython)
        # for row, feat, bin in zip(rows, feats, bins):
        #     print(f"row {row} feat {feat} encoded_data_np[row, feat, bin] {encoded_data_np[row, feat, bin]} encoded_data_cython[row, feat, bin] {encoded_data_cython[row, feat, bin]}, abs diff {np.abs(encoded_data_np[row, feat, bin] - encoded_data_cython[row, feat, bin])}")
        equality_mask = np.isclose(encoded_data_np, encoded_data_cython)
        number_true = np.sum(equality_mask)
        number_false = np.sum(~equality_mask)

        abs_dff = np.abs(encoded_data_np - encoded_data_cython)
        print(f"max diff {np.max(abs_dff)}, number false {number_false:,d}, percent false {number_false / (number_true + number_false) * 100:0.2f}% out of {number_true + number_false:,d} total")

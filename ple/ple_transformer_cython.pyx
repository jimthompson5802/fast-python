# cython: language_level=3

import cython

import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def _ple_transform_cython(np.ndarray[np.float64_t, ndim=1] column_data, np.ndarray[np.float64_t, ndim=1] column_bin_boundaries, int num_bins):

    # Get the number of rows in column_data
    cdef int num_rows = column_data.shape[0]

    # Declare variables i and j for use in loops
    cdef int i, j

    # Initialize a 2D numpy array filled with ones, with dimensions corresponding to the number of data points and the number of bins
    cdef np.ndarray[np.float64_t, ndim=2] encoded_data = np.ones((column_data.shape[0], num_bins))

    # Compute the bin indices for each data point in column_data
    cdef np.ndarray[np.int64_t, ndim=1] bin_indices = np.digitize(column_data, column_bin_boundaries) - 1

    # Compute the minimum value of each bin
    cdef np.ndarray[np.float64_t, ndim=1] bin_min = column_bin_boundaries[bin_indices]

    # Handle edge case where bin index equals the number of bins
    bin_min[bin_indices == num_bins] = column_bin_boundaries[num_bins - 1]

    # Compute the numerator for each data point as the difference between the data point and its corresponding bin_min
    cdef np.ndarray[np.float64_t, ndim=1] bin_numerator = column_data - bin_min

    # Compute the widths of the bins
    cdef np.ndarray[np.float64_t, ndim=1] bin_widths = np.diff(column_bin_boundaries)

    # Create a copy of bin_indices
    cdef np.ndarray[np.int64_t, ndim=1] idxs = bin_indices

    # Adjust for edge case where bin index equals the number of bins
    idxs[idxs == num_bins] = num_bins - 1

    # Compute the denominator for each data point as the corresponding bin width
    cdef np.ndarray[np.float64_t, ndim=1] bin_denominator = bin_widths[idxs]

    # Compute the encoded values for each data point as the ratio of bin_numerator to bin_denominator
    cdef np.ndarray[np.float64_t, ndim=1] encoded_values = bin_numerator / bin_denominator

    # Iterate over the rows of encoded_data to fill in the encoded values
    for i in range(num_rows):
        # Set the value in the mask at the corresponding bin index to the encoded value for that row
        encoded_data[i, bin_indices[i]] = encoded_values[i]
        # Set all subsequent values in the same row to 0
        for j in range(bin_indices[i] + 1, num_bins):
            encoded_data[i, j] = 0

    # Return the transformed encoded_data
    return encoded_data
# cython: language_level=3

import cython

import numpy as np
cimport numpy as np

@cython.boundscheck(False)
def _ple_transform_cython(np.ndarray[np.float64_t, ndim=1] column_data, np.ndarray[np.float64_t, ndim=1] column_bin_boundaries, int num_bins):
    cdef int num_rows = column_data.shape[0]
    cdef int i, row, col

    cdef np.ndarray[np.float64_t, ndim=2] encoded_data = np.ones((column_data.shape[0], num_bins))
    cdef np.ndarray[np.int64_t, ndim=1] bin_indices = np.digitize(column_data, column_bin_boundaries) - 1
    cdef np.ndarray[np.float64_t, ndim=1] bin_min = column_bin_boundaries[bin_indices]
    bin_min[bin_indices == num_bins] = column_bin_boundaries[num_bins - 1]
    cdef np.ndarray[np.float64_t, ndim=1] bin_numerator = column_data - bin_min
    cdef np.ndarray[np.float64_t, ndim=1] bin_widths = np.diff(column_bin_boundaries)
    cdef np.ndarray[np.int64_t, ndim=1] idxs = bin_indices
    idxs[idxs == num_bins] = num_bins - 1
    cdef np.ndarray[np.float64_t, ndim=1] bin_denominator = bin_widths[idxs]
    cdef np.ndarray[np.float64_t, ndim=1] encoded_values = bin_numerator / bin_denominator
    cdef np.ndarray[np.uint8_t, ndim=2] mask = np.zeros((encoded_data.shape[0], encoded_data.shape[1]), dtype=np.bool_)

    for i in range(encoded_data.shape[0]):
        mask[i, bin_indices[i]] = True
    cdef np.ndarray[np.int64_t, ndim=1] rows, cols

    rows, cols = np.where(mask)
    for row, col in zip(rows, cols):
        encoded_data[row, col] = encoded_values[row]
        for i in range(col + 1, encoded_data.shape[1]):
            encoded_data[row, i] = 0
    return encoded_data
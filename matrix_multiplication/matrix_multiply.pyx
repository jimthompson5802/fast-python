# to build extexsion module run:
# cythonize -i matrix_multiply.pyx

#cython: language_level=3

cimport cython
from cython.view cimport array as cvarray
from cython.parallel cimport prange
from libc.stdlib cimport calloc, free
from libc.math cimport isnan


@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_multiply_cp(A, B):
    # Declare integer variables for loop counters and matrix dimensions
    cdef int i, j, k
    cdef int nrows = len(A)  # Number of rows in matrix A
    cdef int ncols = len(B[0])  # Number of columns in matrix B
    cdef int ncols_A = len(A[0])  # Number of columns in matrix A

    # Allocate memory for the result matrix and initialize it with zeros
    cdef double[:,:] result = <double[:nrows, :ncols]> calloc(nrows * ncols, sizeof(double))

    # Create views for the input matrices and the result matrix
    cdef double[:,:] A_view = A
    cdef double[:,:] B_view = B
    cdef double[:,:] result_view = result

    # Perform matrix multiplication
    for i in range(A.shape[0]):  # Iterate over rows of A
        for j in range(B.shape[1]):  # Iterate over columns of B
            # Compute dot product of i-th row of A and j-th column of B
            for k in prange(ncols_A, nogil=True):  
                result[i][j] += A_view[i][k] * B_view[k][j]

    # Return the result matrix
    return result
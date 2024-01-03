# Matrix Muliplication Testbed

## build cython extenions
```bash
cythonize -i matrix_multiplication.pyx
```

## build native c module
```bash
gcc -O3 matrix_multiplication_natice.c -o matrix_multiplication_native
```
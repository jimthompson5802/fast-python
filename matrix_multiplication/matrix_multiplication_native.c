#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 200

// Function to generate a random matrix
void generate_random_matrix(float matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = rand() % 100;  // Fill matrix with random numbers between 0 and 99
        }
    }
}

// Function to multiply two matrices
void multiply_matrices(float a[SIZE][SIZE], float b[SIZE][SIZE], float result[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main() {
    srand(time(0));  // Seed the random number generator

    float matrix1[SIZE][SIZE];
    float matrix2[SIZE][SIZE];
    float result[SIZE][SIZE];

    generate_random_matrix(matrix1);
    generate_random_matrix(matrix2);

    clock_t start = clock();
    multiply_matrices(matrix1, matrix2, result);
    clock_t end = clock();

    double time_taken = ((double)end - start) / CLOCKS_PER_SEC;

    printf("Time taken for matrix multiplication: %f seconds\n", time_taken);

    return 0;
}
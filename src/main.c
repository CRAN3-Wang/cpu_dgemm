#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "utils.h"
#include "my_dgemm.h"

#define EPS (0.005)

void random_matrix(double *matrix, int m, int n) {
    int ldm = m;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i + j * ldm] = (double)drand48();
        }
    }
}

void check(double *C, double *C_ref, int m, int n) {
    int ldc     = m;
    int ldc_ref = m;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fabs(C[i + ldc * j] - C_ref[i + ldc * j]) > EPS) {
                printf("Incorrect at i: %d, j: %d with given: %lf but expect: %lf\n", i, j,
                       C[i + ldc * j], C_ref[i + ldc * j]);
                return;
            }
        }
    }
    printf("Correct\n");
    return;
}

int main() {
    int m = 96;
    int n = 2048;
    int k = 256;

    double *A = (double *)malloc(sizeof(double) * m * k);
    double *B = (double *)malloc(sizeof(double) * k * n);

    // col-major
    int lda = m;
    int ldb = k;

#ifdef DGEMM_MR
    int ldc = ((m - 1) / DGEMM_MR + 1) * DGEMM_MR;
#else
    int ldc = m;
#endif
    int ldc_ref   = m;
    double *C     = aligned_malloc(m, (n + 4), sizeof(double));
    double *C_ref = (double *)malloc(sizeof(double) * m * n);

    random_matrix(A, m, k);
    random_matrix(B, k, n);

    blas_dgemm(A, B, C_ref, m, n, k);
    // naive_dgemm(A, B, C, m, n, k);
    my_dgemm(m, n, k, A, lda, B, ldb, C, ldc);
    check(C, C_ref, m, n);

    // free(A);
    // free(B);
    // free(C);
    // free(C_ref);
    return 0;
}
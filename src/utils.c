#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include "config.h"
void blas_dgemm(double *A, double *B, double *C_ref, int m, int n, int k) {
    double alpha = 1.0;
    double beta  = 0.0;

    int lda = m;
    int ldb = k;
    int ldc = m;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta,
                C_ref, ldc);
}

void naive_dgemm(double *A, double *B, double *C_ref, int m, int n, int k) {
    int lda = m;
    int ldb = k;
    int ldc = m;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            for (int p = 0; p < k; ++p) {
                C_ref[i + j * ldc] += A[i + p * lda] * B[p + j * ldb];
            }
        }
    }
}

double *aligned_malloc(int m, int n, int size) {
    double *ptr;
    int err;

    err = posix_memalign((void **)&ptr, (size_t)GEMM_SIMD_ALIGN_SIZE, size * m * n);

    if (err) {
        printf("bl_malloc_aligned(): posix_memalign() failures");
        exit(1);
    }

    return ptr;
}
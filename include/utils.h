#include <cblas.h>
void blas_dgemm(double *A, double *B, double *C_ref, int m, int n, int k);
void naive_dgemm(double *A, double *B, double *C_ref, int m, int n, int k);
double *aligned_malloc(int m, int n, int size);
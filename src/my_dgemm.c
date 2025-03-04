#include "my_dgemm.h"

#include <stdlib.h>

#include "config.h"
#include "utils.h"
/**
 * @brief This function will pack A to blocks with shape (m, k), col-majored array and store to
 * packA
 *
 * @param m #col of target shape
 * @param k #row of target shape
 * @param XA src matrix
 * @param ldxa leading dimension of XA
 * @param offseta from which part of XA to start packing
 * @param packA dst of packed matrix
 */
static inline void packA_mcxkc_d(int m, int k, double *XA, int ldxa, int offseta, double *packA) {
    int i, p;
    // a_ptr array recording each element in current col, each col has MR entries.
    double *a_ptr[DGEMM_MR];

    // we will first assign first col to each entries in a_ptr.
    for (i = 0; i < m; ++i) {
        a_ptr[i] = XA + (offseta + i);
    }
    // This loop is for senario that m < MR, we use XA + offset to replace this part to avoid mem
    // leak.
    for (i = m; i < DGEMM_MR; ++i) {
        a_ptr[i] = XA + (offseta + 0);
    }

    // The core component of this function, we assign values to packA array from given XA.
    for (p = 0; p < k; ++p) {
        for (i = 0; i < DGEMM_MR; ++i) {
            // assign current col to packA, and move packA to next element.
            *packA = *a_ptr[i];
            packA++;
            // move a_ptr to next i-th row, next col.
            a_ptr[i] = a_ptr[i] + ldxa;
        }
    }
}

/**
 * @brief This function will pack B to blocks with shape (k, n), row-majored array and store to
 * packB
 *
 * @param n #col of target shape
 * @param k #row of target shape
 * @param XB src matrix
 * @param ldxb leading dimension of XB
 * @param offsetb from which part of XB to start packing
 * @param packB dst of packed matrix
 */
static inline void packB_kcxnc_d(int n, int k, double *XB, int ldxb, int offsetb, double *packB) {
    int j, p;
    double *b_ptr[DGEMM_NR];

    for (j = 0; j < n; j++) {
        b_ptr[j] = XB + ldxb * (offsetb + j);
    }

    for (j = n; j < DGEMM_NR; j++) {
        b_ptr[j] = XB + ldxb * (offsetb + 0);
    }

    for (p = 0; p < k; p++) {
        for (j = 0; j < DGEMM_NR; j++) {
            // assign current row to packB, move packB to next place, move b_ptr[j] to next row,
            // current col.
            *packB++ = *b_ptr[j]++;
        }
    }
}

/**
 * @brief The micro kernel of dgemm, this kernel will perform a matmul for input blocks: a(MRxKC)
 * and b(KC, NR).
 *
 * @param k
 * @param a
 * @param b
 * @param c
 * @param ldc
 */
static void micro_kernel(int k, double *a, double *b, double *c, unsigned long long ldc) {
    for (int p = 0; p < k; ++p) {
        for (int j = 0; j < DGEMM_NR; ++j) {
            for (int i = 0; i < DGEMM_MR; ++i) {
                c[i + j * ldc] += a[i + p * DGEMM_MR] * b[p * DGEMM_NR + j];
            }
        }
    }
}

/**
 * @brief The macro kernel of dgemm, this kernel will compute the matmul result of packA and packB.
 * In packA and packB, the memory layout is splitted to smaller blocks A(MRxKC) and B(KC, NR). This
 * function will call micro_kernel to perform the real calculation by iteration, and store back to C
 * as the result of packA(MCxKC) and packB(KCxNC).
 *
 * @param m
 * @param n
 * @param k
 * @param packA
 * @param packB
 * @param C
 * @param ldc
 */
static void macro_kernel(int m, int n, int k, double *packA, double *packB, double *C, int ldc) {
    for (int j = 0; j < n; j += DGEMM_NR) {
        for (int i = 0; i < m; i += DGEMM_MR) {
            micro_kernel(k, &packA[i * k], &packB[j * k], &C[j * ldc + i], (unsigned long long)ldc);
        }
    }
}

void my_dgemm(int m, int n, int k, double *A, int lda, double *B, int ldb, double *C, int ldc) {
    double *packA = aligned_malloc(DGEMM_KC, (DGEMM_MC + 1), sizeof(double));
    double *packB = aligned_malloc(DGEMM_KC, (DGEMM_NC + 1), sizeof(double));
    for (int jc = 0; jc < n; jc += DGEMM_NC) {
        int j_NC = min(n - jc, DGEMM_NC);
        for (int pc = 0; pc < k; pc += DGEMM_KC) {
            int p_KC = min(k - pc, DGEMM_KC);
            for (int j = 0; j < j_NC; j += DGEMM_NR) {
                packB_kcxnc_d(min(j_NC - j, DGEMM_NR), p_KC, &B[pc], ldb, jc + j, &packB[j * p_KC]);
            }
            for (int ic = 0; ic < m; ic += DGEMM_MC) {
                int i_MC = min(m - ic, DGEMM_MC);
                for (int i = 0; i < i_MC; i += DGEMM_MR) {
                    packA_mcxkc_d(min(i_MC - i, DGEMM_MR), p_KC, &A[pc * lda], lda, ic + i,
                                  &packA[i * p_KC]);
                }
                macro_kernel(i_MC, j_NC, p_KC, packA, packB, &C[jc * ldc + ic], ldc);
            }
        }
    }
    free(packA);
    free(packB);
}
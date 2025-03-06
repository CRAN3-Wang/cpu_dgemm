#include <immintrin.h>

void kernel_8x4(const double* A, const double* B, double* C, int k) {
    // We use 2 ymm registers for one col
    // 1st col
    __m256d c_00 = _mm256_load_pd(C + 0 * 8 + 0);
    __m256d c_40 = _mm256_load_pd(C + 0 * 8 + 4);

    // 2nd col
    __m256d c_01 = _mm256_load_pd(C + 1 * 8 + 0);
    __m256d c_41 = _mm256_load_pd(C + 1 * 8 + 4);

    // 3rd col
    __m256d c_02 = _mm256_load_pd(C + 2 * 8 + 0);
    __m256d c_42 = _mm256_load_pd(C + 2 * 8 + 4);

    // 4th col
    __m256d c_03 = _mm256_load_pd(C + 3 * 8 + 0);
    __m256d c_43 = _mm256_load_pd(C + 3 * 8 + 4);

    // Use 2 regs for col of A, 1 reg for row of B
    __m256d a03 = _mm256_setzero_pd();
    __m256d a47 = _mm256_setzero_pd();

    // set a reg to store broadcasted b
    __m256d b_bc = _mm256_setzero_pd();

    for (int i = 0; i < k; ++i) {
        // load the 0-3 row of A at i-th col
        a03 = _mm256_load_pd(A + 8 * i + 0);
        a47 = _mm256_load_pd(A + 8 * i + 4);

        for (int j = 0; j < 4; ++j) {
            // load a element of B from DRAM, broadcast it.
            b_bc = _mm256_broadcast_sd(B + 4 * i + j);

            switch (j) {
                case 0:
                    c_00 = _mm256_fmadd_pd(a03, b_bc, c_00);
                    c_40 = _mm256_fmadd_pd(a47, b_bc, c_40);
                    break;
                case 1:
                    c_01 = _mm256_fmadd_pd(a03, b_bc, c_01);
                    c_41 = _mm256_fmadd_pd(a47, b_bc, c_41);
                    break;
                case 2:
                    c_02 = _mm256_fmadd_pd(a03, b_bc, c_02);
                    c_42 = _mm256_fmadd_pd(a47, b_bc, c_42);
                    break;
                case 3:
                    c_03 = _mm256_fmadd_pd(a03, b_bc, c_03);
                    c_43 = _mm256_fmadd_pd(a47, b_bc, c_43);
                    break;
            }
        }
    }

    // store reg to mem
    _mm256_store_pd(C + 0 * 8 + 0, c_00);
    _mm256_store_pd(C + 0 * 8 + 4, c_40);
    _mm256_store_pd(C + 1 * 8 + 0, c_01);
    _mm256_store_pd(C + 1 * 8 + 4, c_41);
    _mm256_store_pd(C + 2 * 8 + 0, c_02);
    _mm256_store_pd(C + 2 * 8 + 4, c_42);
    _mm256_store_pd(C + 3 * 8 + 0, c_03);
    _mm256_store_pd(C + 3 * 8 + 4, c_43);
}

void kernel_12x4(const double* A, const double* B, double* C, int k) {
    // We use 3 ymm registers for one col
    // 1st col
    __m256d c_00 = _mm256_load_pd(C + 0 * 12 + 0);
    __m256d c_40 = _mm256_load_pd(C + 0 * 12 + 4);
    __m256d c_80 = _mm256_load_pd(C + 0 * 12 + 8);

    // 2nd col
    __m256d c_01 = _mm256_load_pd(C + 1 * 12 + 0);
    __m256d c_41 = _mm256_load_pd(C + 1 * 12 + 4);
    __m256d c_81 = _mm256_load_pd(C + 1 * 12 + 8);

    // 3rd col
    __m256d c_02 = _mm256_load_pd(C + 2 * 12 + 0);
    __m256d c_42 = _mm256_load_pd(C + 2 * 12 + 4);
    __m256d c_82 = _mm256_load_pd(C + 2 * 12 + 8);

    // 4th col
    __m256d c_03 = _mm256_load_pd(C + 3 * 12 + 0);
    __m256d c_43 = _mm256_load_pd(C + 3 * 12 + 4);
    __m256d c_83 = _mm256_load_pd(C + 3 * 12 + 8);

    // Use 3 regs for col of A
    __m256d a03  = _mm256_setzero_pd();
    __m256d a47  = _mm256_setzero_pd();
    __m256d a811 = _mm256_setzero_pd();

    // set a reg to store broadcasted b
    __m256d b_bc = _mm256_setzero_pd();

    for (int i = 0; i < k; ++i) {
        // load the 0-3 row of A at i-th col
        a03  = _mm256_load_pd(A + i * 12 + 0);
        a47  = _mm256_load_pd(A + i * 12 + 4);
        a811 = _mm256_load_pd(A + i * 12 + 8);

        for (int j = 0; j < 4; ++j) {
            // load a element of B from DRAM, broadcast it.
            b_bc = _mm256_broadcast_sd(B + 4 * i + j);

            switch (j) {
                case 0:
                    c_00 = _mm256_fmadd_pd(a03, b_bc, c_00);
                    c_40 = _mm256_fmadd_pd(a47, b_bc, c_40);
                    c_80 = _mm256_fmadd_pd(a811, b_bc, c_80);
                    break;
                case 1:
                    c_01 = _mm256_fmadd_pd(a03, b_bc, c_01);
                    c_41 = _mm256_fmadd_pd(a47, b_bc, c_41);
                    c_81 = _mm256_fmadd_pd(a811, b_bc, c_81);
                    break;
                case 2:
                    c_02 = _mm256_fmadd_pd(a03, b_bc, c_02);
                    c_42 = _mm256_fmadd_pd(a47, b_bc, c_42);
                    c_82 = _mm256_fmadd_pd(a811, b_bc, c_82);
                    break;
                case 3:
                    c_03 = _mm256_fmadd_pd(a03, b_bc, c_03);
                    c_43 = _mm256_fmadd_pd(a47, b_bc, c_43);
                    c_83 = _mm256_fmadd_pd(a811, b_bc, c_83);
                    break;
            }
        }
    }

    // store reg to mem
    _mm256_store_pd(C + 0 * 12 + 0, c_00);
    _mm256_store_pd(C + 0 * 12 + 4, c_40);
    _mm256_store_pd(C + 0 * 12 + 8, c_80);

    _mm256_store_pd(C + 1 * 12 + 0, c_01);
    _mm256_store_pd(C + 1 * 12 + 4, c_41);
    _mm256_store_pd(C + 1 * 12 + 8, c_81);

    _mm256_store_pd(C + 2 * 12 + 0, c_02);
    _mm256_store_pd(C + 2 * 12 + 4, c_42);
    _mm256_store_pd(C + 2 * 12 + 8, c_82);

    _mm256_store_pd(C + 3 * 12 + 0, c_03);
    _mm256_store_pd(C + 3 * 12 + 4, c_43);
    _mm256_store_pd(C + 3 * 12 + 8, c_83);
}
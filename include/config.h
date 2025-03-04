#pragma once
#define GEMM_SIMD_ALIGN_SIZE (32)

#define min(i, j) ((i) < (j) ? (i) : (j))

#define DGEMM_MC 96
#define DGEMM_NC 2048
#define DGEMM_KC 256
#define DGEMM_MR 8
#define DGEMM_NR 4
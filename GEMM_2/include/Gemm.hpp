
#pragma once
#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <cstring>
#include <iostream>

template <typename T>
void matrixMul(size_t m, size_t n, size_t K, T* a, T* b, T* c) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            T sum = 0;
            for (size_t k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

template <typename T>
void matrixMul_looptransform(size_t m, size_t n, size_t K, T* a, T* b, T* c) {
    memset(c, 0, m * n * sizeof(T));
    for (size_t i = 0; i < m; i++) {
        for (size_t k = 0; k < K; k++) {
            for (size_t j = 0; j < n; j++) {
                c[i * n + j] += a[i * K + k] * b[k * n + j];
            }
        }
    }
}

template <size_t BM, size_t BK, size_t BN, typename T>
void matrixMul_avx(T* a, T* b, T* c) {
    static_assert(BN % 8 == 0, "BN must be multiple of 8");
    static_assert(std::is_same_v<T, float>, "Only float is supported");

    constexpr size_t B_REG_M = 2;
    __m256 c_vec[B_REG_M * (BN / 8)];

    for (int i = 0; i < BM; i += B_REG_M) {
        for (int k = 0; k < B_REG_M * (BN / 8); k++) {
            c_vec[k] = _mm256_setzero_ps();
        }

        for (int k = 0; k < BK; k++) {
            __m256 b_vec[BN / 8];

            for (int jj = 0; jj < BN / 8; jj++) {
                b_vec[jj] = _mm256_load_ps(b + k * BN + jj * 8);
            }

            for (int ii = 0; ii < B_REG_M; ii++) {
                __m256 a_vec = _mm256_broadcast_ss(a + (i + ii) * BK + k);

                for (int jj = 0; jj < BN / 8; jj++) {
                    c_vec[ii * (BN / 8) + jj] = _mm256_fmadd_ps(
                        a_vec, b_vec[jj], c_vec[ii * (BN / 8) + jj]);
                }
            }
        }

        for (int ii = 0; ii < B_REG_M; ii++) {
            for (int jj = 0; jj < BN / 8; jj++) {
                _mm256_store_ps(c + (i + ii) * BN + jj * 8,
                                c_vec[ii * (BN / 8) + jj]);
            }
        }
    }
}

#pragma once
#include <assert.h>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <stddef.h>
#include <utility>
template <size_t filterSize, class T>
void Convolution(int imageInSizeX, int imageInSizeY, const T *imageIn,
                 const T *filter, T *imageOut) {
  int imageOutSizeX = imageInSizeX - filterSize + 1;
  int imageOutSizeY = imageInSizeY - filterSize + 1;

  for (int y = 0; y < imageOutSizeY; y++) {
    for (int x = 0; x < imageOutSizeX; x++) {
      T sum = 0;
#pragma unroll
      for (int fy = 0; fy < filterSize; fy++) {
#pragma unroll
        for (int fx = 0; fx < filterSize; fx++) {
          auto filterItem = filter[fy * filterSize + fx];
          auto ImageItem = imageIn[(fy + y) * imageInSizeX + (fx + x)];
          sum += filterItem * ImageItem;
        }
      }
      imageOut[y * imageOutSizeX + x] = sum;
    }
  }
  return;
}

template <int INDEX, int BX, int BY, class FUNC>
constexpr void loop_unrool(FUNC func) [[gnu::always_inline]] {
  if constexpr (INDEX <= BX * BY - 1) {
    constexpr int i = INDEX % BX;
    constexpr int j = INDEX / BX;
    func(i, j);
  }
  if constexpr (INDEX + 1 <= BX * BY - 1)
    loop_unrool<INDEX + 1, BX, BY>(func);
}

template <size_t filterSize, int BX, int BY, class T>
void Convolution_unroll(int imageInSizeX, int imageInSizeY, T *imageIn,
                        T *filter, T *imageOut) {
  int imageOutSizeX = imageInSizeX - filterSize + 1;
  int imageOutSizeY = imageInSizeY - filterSize + 1;

  assert(imageOutSizeX % BX == 0);
  assert(imageOutSizeY % BY == 0);
  assert(filterSize % 2 == 1);

  for (int y = 0; y < imageOutSizeY; y += BY) {
    for (int x = 0; x < imageOutSizeX; x += BX) {
      T sum[BX * BY] = {0};
      for (int fy = 0; fy < filterSize; fy++) {
        for (int fx = 0; fx < filterSize; fx++) {
          T filterItem = filter[fy * filterSize + fx];
          auto lam = [&](int i, int j) [[gnu::always_inline]] {
            auto ImageItem =
                imageIn[(fy + y + j) * imageInSizeX + (fx + x + i)];
            sum[j * BX + i] += filterItem * ImageItem;
          };
          loop_unrool<0, BX, BY>(lam);
        }
      }
      auto lam = [&](int i, int j) [[gnu::always_inline]] {
        imageOut[(y + j) * imageOutSizeX + (x + i)] = sum[j * BX + i];
      };
      loop_unrool<0, BX, BY>(lam);
    }
  }
  return;
}

template <size_t filterSize, int BX, int BY, class T>
void Convolution_unroll_ref(int imageInSizeX, int imageInSizeY, T *imageIn,
                            T *filter, T *imageOut) {
  int imageOutSizeX = imageInSizeX - filterSize + 1;
  int imageOutSizeY = imageInSizeY - filterSize + 1;

  assert(imageOutSizeX % BX == 0);
  assert(imageOutSizeY % BY == 0);
  assert(filterSize % 2 == 1);

  for (int y = 0; y < imageOutSizeY; y += BY) {
    for (int x = 0; x < imageOutSizeX; x += BX) {
      T sum[BX * BY] = {0};
#pragma unroll
      for (int fy = 0; fy < filterSize; fy++) {
#pragma unroll
        for (int fx = 0; fx < filterSize; fx++) {
          T filterItem = filter[fy * filterSize + fx];
#pragma unroll
          for (int i = 0; i < BY; i++) {
#pragma unroll
            for (int j = 0; j < BX; j++) {
              auto ImageItem =
                  imageIn[(fy + y + i) * imageInSizeX + (fx + x + j)];
              sum[i * BX + j] += filterItem * ImageItem;
            }
          }
        }
      }
#pragma unroll
      for (int i = 0; i < BY; i++) {
#pragma unroll
        for (int j = 0; j < BX; j++) {
          imageOut[(y + i) * imageOutSizeX + (x + j)] = sum[i * BX + j];
        }
      }
    }
  }
  return;
}

template <size_t filterSize, int BX, int BY, class T>
void Convolution_unroll_simd_avx(int imageInSizeX, int imageInSizeY, T *imageIn,
                                 T *filter, T *imageOut) {
  int imageOutSizeX = imageInSizeX - filterSize + 1;
  int imageOutSizeY = imageInSizeY - filterSize + 1;

  assert(imageOutSizeX % (BX * 8) == 0);
  assert(imageOutSizeY % BY == 0);
  assert(filterSize % 2 == 1);

#pragma omp parallel for
  for (int y = 0; y < imageOutSizeY; y += BY) {
    for (int x = 0; x < imageOutSizeX; x += BX * 8) {

      __m256i sum[BY][BX] = {_mm256_setzero_si256()};
      for (int fy = 0; fy < filterSize; fy++) {
        for (int fx = 0; fx < filterSize; fx++) {

          __m256i filterItem = _mm256_set1_epi32(filter[fy * filterSize + fx]);
          auto lam = [&](int i, int j) [[gnu::always_inline]] {
            __m256i ImageItem = _mm256_loadu_si256(
                (__m256i *)(imageIn + (fy + y + j) * imageInSizeX +
                            (fx + x + i * 8)));
            sum[j][i] = _mm256_add_epi32(
                sum[j][i], _mm256_mullo_epi32(filterItem, ImageItem));
          };
          loop_unrool<0, BX, BY>(lam);
        }
      }
      auto lam = [&](int i, int j) [[gnu::always_inline]] {
        _mm256_storeu_si256(
            (__m256i *)(imageOut + (y + j) * imageOutSizeX + (x + i * 8)),
            sum[j][i]);
      };
      loop_unrool<0, BX, BY>(lam);
    }
  }
  return;
}

template <size_t filterSize, int BX, int BY, class T>
void Convolution_unroll_simd_avx512(int imageInSizeX, int imageInSizeY,
                                    T *imageIn, T *filter, T *imageOut) {
  int imageOutSizeX = imageInSizeX - filterSize + 1;
  int imageOutSizeY = imageInSizeY - filterSize + 1;

  assert(imageOutSizeX % (BX * 16) == 0);
  assert(imageOutSizeY % BY == 0);
  assert(filterSize % 2 == 1);

#pragma omp parallel for
  for (int y = 0; y < imageOutSizeY; y += BY) {
    for (int x = 0; x < imageOutSizeX; x += BX * 16) {

      __m512i sum[BY][BX] = {_mm512_setzero_si512()};
      for (int fy = 0; fy < filterSize; fy++) {
        for (int fx = 0; fx < filterSize; fx++) {

          __m512i filterItem = _mm512_set1_epi32(filter[fy * filterSize + fx]);
          auto lam = [&](int i, int j) [[gnu::always_inline]] {
            __m512i ImageItem = _mm512_loadu_si512(
                (__m512i *)(imageIn + (fy + y + j) * imageInSizeX +
                            (fx + x + i * 16)));
            sum[j][i] = _mm512_add_epi32(
                sum[j][i], _mm512_mullo_epi32(filterItem, ImageItem));
          };
          loop_unrool<0, BX, BY>(lam);
        }
      }
      auto lam = [&](int i, int j) [[gnu::always_inline]] {
        _mm512_storeu_si512(
            (__m512i *)(imageOut + (y + j) * imageOutSizeX + (x + i * 16)),
            sum[j][i]);
      };
      loop_unrool<0, BX, BY>(lam);
    }
  }
  return;
}
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 定义矩阵大小
#define M 1024*8
#define N 1024*8
#define K 1024*8

// 定义计时器
float milliseconds = 0;

// CUDA错误检查宏
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

int main() {
    // 分配内存和初始化数据
    float *h_A, *h_B, *h_C;
    h_A = new float[M * K];
    h_B = new float[K * N];
    h_C = new float[M * N];

    for (int i = 0; i < M * K; ++i)
        h_A[i] = i;

    for (int i = 0; i < K * N; ++i)
        h_B[i] = i;

    // 初始化CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 在设备上分配内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

    // 将数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 创建CUDA事件
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 启动计时器
    CUDA_CHECK(cudaEventRecord(start));

    // 执行矩阵乘法
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);

    // 停止计时器
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // 将结果从设备复制回主机
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 打印结果
    //std::cout << "Result Matrix:" << std::endl;
    //for (int i = 0; i < M; ++i) {
     //   for (int j = 0; j < N; ++j) {
    //        std::cout << h_C[i * N + j] << " ";
    //    }
    //    std::cout << std::endl;
   // }

    // 打印计时结果
    std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;

    // 释放内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    cublasDestroy(handle);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
      

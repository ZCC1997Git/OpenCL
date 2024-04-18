/*
 *  the native version of GEMM
 */
template <class T>
__kernel void matrixMultiplyNativeKernel(__global T* A,
                                         __global T* B,
                                         __global T* C,
                                         int M,
                                         int N,
                                         int K) {
    int i = get_global_id(1);
    int j = get_global_id(0);

    T sum = 0;

    for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
    }

    C[i * N + j] = sum;
}
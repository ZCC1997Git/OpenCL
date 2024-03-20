/*
* Gemm on GPU
*/

__kernel void ClGemm(__global float* A, __global float* B, __global float* C, uint height, uint width) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;
    for (int i = 0; i < width; i++) {
        sum += A[row * width + i] * B[i * width + col];
    }
    C[row * width + col] = sum;
}
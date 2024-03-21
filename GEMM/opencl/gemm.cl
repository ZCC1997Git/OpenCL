/*
* Gemm on GPU
*/
__kernel void ClGemm(__global T* A, __global T* B, __global T* C, uint height, uint width) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    T sum = 0.0;
    for (int i = 0; i < width; i++) {
        sum += A[row * width + i] * B[i * width + col];
    }
    C[row * width + col] = sum;
}
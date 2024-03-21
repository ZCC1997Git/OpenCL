/*
 * Gemm on GPU
 */
#define tile_size 32
__kernel void ClGemm(__global T* A,
                     __global T* B,
                     __global T* C,
                     uint height,
                     uint width) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    T sum = 0.0;
    for (int i = 0; i < width; i++) {
        sum += A[row * width + i] * B[i * width + col];
    }
    C[row * width + col] = sum;
}

__kernel void ClGemm_block(__global T* A,
                           __global T* B,
                           __global T* C,
                           uint height,
                           uint width) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    int local_y = get_local_id(0);
    int local_x = get_local_id(1);

    __local T local_A[tile_size][tile_size];
    __local T local_B[tile_size][tile_size];

    int num_iter = width / tile_size;

    __private T sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    for (int i = 0; i < num_iter; i++) {
        local_A[local_x][local_y] = A[row * width + i * tile_size + local_y];
        local_B[local_x][local_y] = B[(i * tile_size + local_x) * width + col];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < tile_size; k += 4) {
            sum0 += local_A[local_x][k] * local_B[k][local_y];
            sum1 += local_A[local_x][k + 1] * local_B[k + 1][local_y];
            sum2 += local_A[local_x][k + 2] * local_B[k + 2][local_y];
            sum3 += local_A[local_x][k + 3] * local_B[k + 3][local_y];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    /*write to global memory*/
    C[row * width + col] = sum0 + sum1 + sum2 + sum3;
}

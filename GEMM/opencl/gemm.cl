/*
 * Gemm on GPU
 */
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

#define tile_size 32
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

#define register_block_size 4
__kernel void ClGemm_block_newversion(__global float* A,
                                      __global float* B,
                                      __global float* C,
                                      uint height,
                                      uint width) {
    int global_y = get_global_id(0);
    int global_x = get_global_id(1);

    int local_y = get_local_id(0);
    int local_x = get_local_id(1);

    /* local memory block*/
    const int vector_tile_size = tile_size / 4;
    __local float4 local_A[tile_size][vector_tile_size];
    __local float4 local_B[tile_size][vector_tile_size];

    /*rgeister block */
    __private float4 register_A[register_block_size];
    __private float4 register_B[register_block_size];
    __private float4 register_C[4] = {
        {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    int num_iter = width / tile_size;

    __private float sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    float4* A_ptr = (__global float4*)A;
    float4* B_ptr = (__global float4*)B;
    float4* C_ptr = (__global float4*)C;

    int new_width = width / 4;
    int strid_x = tile_size / get_local_size(1);

    for (int i = 0; i < num_iter; i++) {
        /*load to local*/
        for (int j = 0; j < strid_x; j++) {
            local_A[strid_x * local_x + j][local_y] =
                A_ptr[strid_x * global_x * new_width + j * new_width +
                      i * vector_tile_size + local_y];
            local_B[strid_x * local_x + j][local_y] =
                B_ptr[tile_size * i * new_width +
                      (strid_x * local_x + j) * new_width + global_y];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < tile_size / 4; j++) {
            /*load to register*/
            for (int k = 0; k < register_block_size; k++) {
                register_A[k] = local_A[local_x * strid_x + k][j];
                register_B[k] = local_B[j * strid_x + k][local_y];
            }

            /*calculate the matrix mul martix*/
            for (int k = 0; k < register_block_size; k++) {
                register_C[k].x += register_A[k].x * register_B[0].x +
                                   register_A[k].y * register_B[1].x +
                                   register_A[k].z * register_B[2].x +
                                   register_A[k].w * register_B[3].x;
                register_C[k].y += register_A[k].x * register_B[0].y +
                                   register_A[k].y * register_B[1].y +
                                   register_A[k].z * register_B[2].y +
                                   register_A[k].w * register_B[3].y;
                register_C[k].z += register_A[k].x * register_B[0].z +
                                   register_A[k].y * register_B[1].z +
                                   register_A[k].z * register_B[2].z +
                                   register_A[k].w * register_B[3].z;
                register_C[k].w += register_A[k].x * register_B[0].w +
                                   register_A[k].y * register_B[1].w +
                                   register_A[k].z * register_B[2].w +
                                   register_A[k].w * register_B[3].w;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /*write to global memory*/
    for (int i = 0; i < 4; i++) {
        C_ptr[(global_x * 4 + i) * new_width + global_y] = register_C[i];
    }
}
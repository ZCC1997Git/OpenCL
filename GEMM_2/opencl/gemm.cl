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

template <int BS>
__kernel void matrixMultiplyBlockKernel(__global float* A,
                                        __global float* B,
                                        __global float* C,
                                        int M,
                                        int N,
                                        int K) {
    int by = get_group_id(1);
    int bx = get_group_id(0);
    int ty = get_local_id(1);
    int tx = get_local_id(0);

    local float ta[BS][BS];
    local float tb[BS][BS];

    int ab = K * BS * by;
    int ae = ab + K;

    int bb = BS * bx;
    float v = 0.0f;

    int i, j;
    for (i = ab, j = bb; i < ae; i += BS, j += BS) {
        ta[ty][tx] = A[i + K * ty + tx];
        tb[ty][tx] = B[j + N * ty + tx];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < BS; k++) {
            v += ta[ty][k] * tb[k][tx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[BS * N * by + BS * bx + N * ty + tx] = v;
}

template <int BS>
__kernel void matrixMultiplyVectorKernel(__global float* A,
                                         __global float* B,
                                         __global float* C,
                                         int M,
                                         int N,
                                         int K) {
    int by = get_group_id(1);
    int bx = get_group_id(0);
    int ty = get_local_id(1);
    int tx = get_local_id(0);

    float4* BB = (float4*)B;
    float4* CC = (float4*)C;

    local float4 ta[BS][BS];
    local float4 tb[BS][BS];

    int ab = 4 * K * BS * by;
    int ae = ab + K;

    int bb = BS * bx;

    float4 v[4];
    v[0] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    v[1] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    v[2] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    v[3] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    int N_float4 = N / 4;
    int i, j;

    for (i = ab, j = bb; i < ae; i += BS, j += BS * N_float4) {
        float4 temp;
        temp.x = A[0 * BS * K + i + K * ty + tx];
        temp.y = A[1 * BS * K + i + K * ty + tx];
        temp.z = A[2 * BS * K + i + K * ty + tx];
        temp.w = A[3 * BS * K + i + K * ty + tx];
        ta[ty][tx] = temp;
        tb[ty][tx] = BB[j + N_float4 * ty + tx];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < BS; k++) {
            v[0] += ta[ty][k].x * tb[k][tx];
            v[1] += ta[ty][k].y * tb[k][tx];
            v[2] += ta[ty][k].z * tb[k][tx];
            v[3] += ta[ty][k].w * tb[k][tx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int ii = 0; ii < 4; ii++) {
        CC[N_float4 * (BS * (ii + by * 4) + ty) + bx * BS + tx] = v[ii];
    }
}

template <int BS>
__kernel void matrixMultiplyVectorMultiItemKernel(__global float* A,
                                                  __global float* B,
                                                  __global float* C,
                                                  int M,
                                                  int N,
                                                  int K) {
    int by = get_group_id(1);
    int bx = get_group_id(0);
    int ty = get_local_id(1);
    int tx = get_local_id(0);

    float4* BB = (float4*)B;
    float4* CC = (float4*)C;

#define unroll_m 8
#define unroll_n 8
#define unroll_m_float4 (unroll_m / 4)
#define unroll_n_float4 (unroll_n / 4)

    local float4 ta[BS * unroll_m_float4][BS];
    local float4 tb[BS * unroll_n_float4][BS];

    int ab = unroll_m * K * BS * by;
    int ae = ab + K;

    int bb = BS * bx * unroll_n_float4;

    float4 v[unroll_m][unroll_n_float4];
    for (int i = 0; i < unroll_m; i++) {
        for (int j = 0; j < unroll_n_float4; j++) {
            v[i][j] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    int N_float4 = N / 4;
    int i, j;

    for (i = ab, j = bb; i < ae; i += BS, j += BS * N_float4) {
        for (int ii = 0; ii < unroll_m_float4; ii++) {
            float4 temp;
            temp.x = A[(4 * ii + 0) * BS * K + i + K * ty + tx];
            temp.y = A[(4 * ii + 1) * BS * K + i + K * ty + tx];
            temp.z = A[(4 * ii + 2) * BS * K + i + K * ty + tx];
            temp.w = A[(4 * ii + 3) * BS * K + i + K * ty + tx];
            ta[ii * BS + ty][tx] = temp;
        }

        for (int jj = 0; jj < unroll_n_float4; jj++) {
            tb[jj * BS + ty][tx] = BB[j + N_float4 * ty + jj * BS + tx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < BS; k++) {
            for (int ii = 0; ii < unroll_m_float4; ii++) {
                for (int jj = 0; jj < unroll_n_float4; jj++) {
                    float4 temp_a = ta[ii * BS + ty][k];
                    float4 temp_b = tb[jj * BS + k][tx];
                    v[4 * ii + 0][jj] += temp_a.x * temp_b;
                    v[4 * ii + 1][jj] += temp_a.y * temp_b;
                    v[4 * ii + 2][jj] += temp_a.z * temp_b;
                    v[4 * ii + 3][jj] += temp_a.w * temp_b;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int ii = 0; ii < unroll_m; ii++) {
        for (int jj = 0; jj < unroll_n_float4; jj++) {
            CC[N_float4 * (BS * (ii + by * unroll_m) + ty) +
               bx * BS * unroll_n_float4 + jj * BS + tx] = v[ii][jj];
        }
    }
}

template <int BS>
__kernel void matrixMultiplyVectorKernel_prefetch(__global float* A,
                                                  __global float* B,
                                                  __global float* C,
                                                  int M,
                                                  int N,
                                                  int K) {
    int by = get_group_id(1);
    int bx = get_group_id(0);
    int ty = get_local_id(1);
    int tx = get_local_id(0);

    float4* BB = (float4*)B;
    float4* CC = (float4*)C;

    local float4 ta[BS][BS];
    local float4 tb[BS][BS];

    int ab = 4 * K * BS * by;
    int ae = ab + K;

    int bb = BS * bx;

    float4 v[4];
    v[0] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    v[1] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    v[2] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    v[3] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    int N_float4 = N / 4;
    int i, j;
    float4 temp_A;
    float4 temp_B;
    temp_A.x = A[0 * BS * K + ab + K * ty + tx];
    temp_A.y = A[1 * BS * K + ab + K * ty + tx];
    temp_A.z = A[2 * BS * K + ab + K * ty + tx];
    temp_A.w = A[3 * BS * K + ab + K * ty + tx];
    temp_B = BB[bb + N_float4 * ty + tx];

    for (i = ab, j = bb; i < ae - BS; i += BS, j += BS * N_float4) {
        barrier(CLK_LOCAL_MEM_FENCE);
        ta[ty][tx] = temp_A;
        tb[ty][tx] = temp_B;
        barrier(CLK_LOCAL_MEM_FENCE);
        temp_A.x = A[0 * BS * K + i + BS + K * ty + tx];
        temp_A.y = A[1 * BS * K + i + BS + K * ty + tx];
        temp_A.z = A[2 * BS * K + i + BS + K * ty + tx];
        temp_A.w = A[3 * BS * K + i + BS + K * ty + tx];
        temp_B = BB[j + BS * N_float4 + N_float4 * ty + tx];
#pragma unroll
        for (int k = 0; k < BS; k++) {
            v[0] += ta[ty][k].x * tb[k][tx];
            v[1] += ta[ty][k].y * tb[k][tx];
            v[2] += ta[ty][k].z * tb[k][tx];
            v[3] += ta[ty][k].w * tb[k][tx];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    ta[ty][tx] = temp_A;
    tb[ty][tx] = temp_B;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < BS; k++) {
        v[0] += ta[ty][k].x * tb[k][tx];
        v[1] += ta[ty][k].y * tb[k][tx];
        v[2] += ta[ty][k].z * tb[k][tx];
        v[3] += ta[ty][k].w * tb[k][tx];
    }

    for (int ii = 0; ii < 4; ii++) {
        CC[N_float4 * (BS * (ii + by * 4) + ty) + bx * BS + tx] = v[ii];
    }
}
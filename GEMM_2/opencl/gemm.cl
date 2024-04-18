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

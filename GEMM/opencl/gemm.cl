/*
* Gemm on GPU
*/
#define tile_size 32
__kernel void ClGemm(__global T* A, __global T* B, __global T* C, uint height, uint width) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    T sum = 0.0;
    for (int i = 0; i < width; i++) {
        sum += A[row * width + i] * B[i * width + col];
    }
    C[row * width + col] = sum;
}


__kernel void ClGemm_block(__global T* A, __global T* B, __global T* C, uint height, uint width) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    int local_x= get_local_id(0);
    int local_y= get_local_id(1);

    __local T local_A[tile_size][tile_size];
    __local T local_B[tile_size][tile_size];
    __local T local_C[tile_size][tile_size];

    /*initialize local_C*/
    local_C[local_x][local_y]=0.0f;
    
    int num_iter= width/tile_size;

    for(int i=0; i<num_iter; i++){
        local_A[local_x][local_y]=A[row*width+i*tile_size+local_y];
        local_B[local_x][local_y]=B[(i*tile_size+local_x)*width+col];
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int k=0;k<tile_size;k++)
        {
            local_C[local_x][local_y]+=local_A[local_x][k]*local_B[k][local_y];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        /*write to global memory*/
        C[row*width+col]=local_C[local_x][local_y];
    }
}

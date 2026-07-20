#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void local_softmax_kernel(const float* input, float* partial_maxs, float* partial_sums, int N) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ float sdata[];
    float *shared_max = sdata;
    float *shared_sum = sdata + blockDim.x;

    float value = idx < N ? input[idx] : -FLT_MAX;
    shared_max[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid+stride]);
        }
        __syncthreads();
    }
    float block_max = shared_max[0];
    float exp_value = idx < N ? expf(value-block_max) : 0.0f;
    shared_sum[tid] = exp_value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid+stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_maxs[blockIdx.x] = block_max;
        partial_sums[blockIdx.x] = shared_sum[0];
    }
    
}

__global__ void global_softmax_kernel(const float* input, const float* partial_maxs, const float* partial_sums, 
    float* output, int N, int num_blocks) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ float sdata[];
    float *shared_max = sdata;
    float *shared_sum = sdata + blockDim.x;

    float local_max = -FLT_MAX;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        local_max = fmaxf(local_max, partial_maxs[i]);
    }
    shared_max[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid+stride]);
        }
        __syncthreads();
    }
    float global_max = shared_max[0];

    float local_sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        local_sum = local_sum + partial_sums[i] * expf(partial_maxs[i] - global_max);
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid+stride];
        }
        __syncthreads();
    }
    float global_sum = shared_sum[0];

    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - global_max) / global_sum;
    }

}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *partial_maxs = NULL;
    float *partial_sums = NULL;
    cudaMalloc(&partial_maxs, blocksPerGrid * sizeof(float));
    cudaMalloc(&partial_sums, blocksPerGrid * sizeof(float));

    size_t shared_memory_size = 2 * threadsPerBlock * sizeof(float);

    local_softmax_kernel<<<blocksPerGrid, threadsPerBlock, shared_memory_size>>>(input, partial_maxs, partial_sums, N);
    global_softmax_kernel<<<1, threadsPerBlock, shared_memory_size>>>(input, partial_maxs, partial_sums, output, N, blocksPerGrid);
    cudaDeviceSynchronize();
    cudaFree(partial_maxs);
    cudaFree(partial_sums);
}

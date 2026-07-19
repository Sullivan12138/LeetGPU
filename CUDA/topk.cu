#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 4096;
constexpr float NEG_INF = -3.4028234663852886e+38F;

__device__ bool better_candidate(float lhs_value, int lhs_index, float rhs_value, int rhs_index) {
    if (lhs_index < 0) {
        return false;
    }
    if (rhs_index < 0) {
        return true;
    }
    return lhs_value > rhs_value || (lhs_value == rhs_value && lhs_index < rhs_index);
}

__global__ void find_block_best_kernel(const float* __restrict__ input,
                                       const unsigned char* __restrict__ selected,
                                       float* __restrict__ block_values,
                                       int* __restrict__ block_indices,
                                       int N) {
    __shared__ float values[THREADS];
    __shared__ int indices[THREADS];

    float best_value = NEG_INF;
    int best_index = -1;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * blockDim.x) {
        if (selected[i] == 0) {
            const float value = input[i];
            if (better_candidate(value, i, best_value, best_index)) {
                best_value = value;
                best_index = i;
            }
        }
    }

    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride &&
            better_candidate(values[threadIdx.x + stride],
                             indices[threadIdx.x + stride],
                             values[threadIdx.x],
                             indices[threadIdx.x])) {
            values[threadIdx.x] = values[threadIdx.x + stride];
            indices[threadIdx.x] = indices[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_values[blockIdx.x] = values[0];
        block_indices[blockIdx.x] = indices[0];
    }
}

__global__ void find_global_best_kernel(float* __restrict__ block_values,
                                        int* __restrict__ block_indices,
                                        int num_blocks) {
    __shared__ float values[THREADS];
    __shared__ int indices[THREADS];

    float best_value = NEG_INF;
    int best_index = -1;

    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        const float value = block_values[i];
        const int index = block_indices[i];
        if (better_candidate(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }

    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride &&
            better_candidate(values[threadIdx.x + stride],
                             indices[threadIdx.x + stride],
                             values[threadIdx.x],
                             indices[threadIdx.x])) {
            values[threadIdx.x] = values[threadIdx.x + stride];
            indices[threadIdx.x] = indices[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_values[0] = values[0];
        block_indices[0] = indices[0];
    }
}

__global__ void write_pick_kernel(float* __restrict__ output,
                                  unsigned char* __restrict__ selected,
                                  const float* __restrict__ block_values,
                                  const int* __restrict__ block_indices,
                                  int rank,
                                  int N) {
    const int index = block_indices[0];
    if (index >= 0 && index < N) {
        output[rank] = block_values[0];
        selected[index] = 1;
    }
}

__global__ void fill_tail_kernel(float* output, int begin, int k) {
    const int i = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        output[i] = NEG_INF;
    }
}

}  // namespace

// input and output are device pointers.
extern "C" void solve(const float* input, float* output, int N, int k) {
    if (N <= 0 || k <= 0) {
        return;
    }

    const int picks = (k < N) ? k : N;
    const int num_blocks_for_n = (N + THREADS - 1) / THREADS;
    const int num_blocks = (num_blocks_for_n < MAX_BLOCKS) ? num_blocks_for_n : MAX_BLOCKS;

    unsigned char* selected = nullptr;
    float* block_values = nullptr;
    int* block_indices = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&selected), static_cast<size_t>(N) * sizeof(unsigned char));
    cudaMalloc(reinterpret_cast<void**>(&block_values), static_cast<size_t>(num_blocks) * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&block_indices), static_cast<size_t>(num_blocks) * sizeof(int));
    cudaMemset(selected, 0, static_cast<size_t>(N) * sizeof(unsigned char));

    for (int rank = 0; rank < picks; ++rank) {
        find_block_best_kernel<<<num_blocks, THREADS>>>(input, selected, block_values, block_indices, N);
        find_global_best_kernel<<<1, THREADS>>>(block_values, block_indices, num_blocks);
        write_pick_kernel<<<1, 1>>>(output, selected, block_values, block_indices, rank, N);
    }

    if (k > N) {
        const int tail = k - N;
        const int blocks = (tail + THREADS - 1) / THREADS;
        fill_tail_kernel<<<blocks, THREADS>>>(output, N, k);
    }

    cudaFree(selected);
    cudaFree(block_values);
    cudaFree(block_indices);
}

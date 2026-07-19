#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 256;
constexpr int TILE_K = 16;
constexpr int TILE_ROWS = 8;

__global__ void count_row_nnz_kernel(const float* __restrict__ A,
                                     int* __restrict__ row_counts,
                                     int M,
                                     int N) {
    __shared__ int counts[THREADS];

    const int row = blockIdx.x;
    int count = 0;

    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        if (A[row * N + col] != 0.0f) {
            ++count;
        }
    }

    counts[threadIdx.x] = count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            counts[threadIdx.x] += counts[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        row_counts[row] = counts[0];
    }
}

__global__ void exclusive_scan_rows_kernel(const int* __restrict__ row_counts,
                                           int* __restrict__ row_offsets,
                                           int M) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int running = 0;
        row_offsets[0] = 0;
        for (int row = 0; row < M; ++row) {
            running += row_counts[row];
            row_offsets[row + 1] = running;
        }
    }
}

__global__ void fill_csr_kernel(const float* __restrict__ A,
                                const int* __restrict__ row_offsets,
                                int* __restrict__ row_write_counts,
                                int* __restrict__ col_indices,
                                float* __restrict__ values,
                                int M,
                                int N,
                                int nnz) {
    const int row = blockIdx.x;

    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        const float value = A[row * N + col];
        if (value != 0.0f) {
            const int offset = atomicAdd(&row_write_counts[row], 1);
            const int pos = row_offsets[row] + offset;
            if (pos < nnz) {
                col_indices[pos] = col;
                values[pos] = value;
            }
        }
    }
}

__global__ void csr_spmm_kernel(const int* __restrict__ row_offsets,
                                const int* __restrict__ col_indices,
                                const float* __restrict__ values,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M,
                                int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_ROWS + ty;
    const int k = blockIdx.x * TILE_K + tx;

    if (row >= M || k >= K) {
        return;
    }

    float acc = 0.0f;
    const int begin = row_offsets[row];
    const int end = row_offsets[row + 1];

    for (int p = begin; p < end; ++p) {
        const int col = col_indices[p];
        acc += values[p] * B[col * K + k];
    }

    C[row * K + k] = acc;
}

}  // namespace

// A, B, and C are device pointers.
// A: [M, N] sparse matrix stored densely in row-major order.
// B: [N, K] dense matrix stored in row-major order.
// C: [M, K] dense output matrix stored in row-major order.
extern "C" void solve(const float* A,
                      const float* B,
                      float* C,
                      int M,
                      int N,
                      int K,
                      int nnz) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    if (nnz <= 0) {
        cudaMemset(C, 0, static_cast<size_t>(M) * K * sizeof(float));
        return;
    }

    int* row_counts = nullptr;
    int* row_offsets = nullptr;
    int* row_write_counts = nullptr;
    int* col_indices = nullptr;
    float* values = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&row_counts), static_cast<size_t>(M) * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&row_offsets), static_cast<size_t>(M + 1) * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&row_write_counts), static_cast<size_t>(M) * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&col_indices), static_cast<size_t>(nnz) * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&values), static_cast<size_t>(nnz) * sizeof(float));

    count_row_nnz_kernel<<<M, THREADS>>>(A, row_counts, M, N);
    exclusive_scan_rows_kernel<<<1, 1>>>(row_counts, row_offsets, M);
    cudaMemset(row_write_counts, 0, static_cast<size_t>(M) * sizeof(int));
    fill_csr_kernel<<<M, THREADS>>>(A, row_offsets, row_write_counts, col_indices, values, M, N, nnz);

    dim3 block(TILE_K, TILE_ROWS);
    dim3 grid((K + TILE_K - 1) / TILE_K, (M + TILE_ROWS - 1) / TILE_ROWS);
    csr_spmm_kernel<<<grid, block>>>(row_offsets, col_indices, values, B, C, M, K);

    cudaFree(row_counts);
    cudaFree(row_offsets);
    cudaFree(row_write_counts);
    cudaFree(col_indices);
    cudaFree(values);
}

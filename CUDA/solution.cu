#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int TILE = 16;

__global__ void gemm_kernel(const half* __restrict__ A,
                            const half* __restrict__ B,
                            half* __restrict__ C,
                            int M,
                            int N,
                            int K,
                            float alpha,
                            float beta) {
    __shared__ half As[TILE][TILE + 1];
    __shared__ half Bs[TILE][TILE + 1];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int tile_k = 0; tile_k < K; tile_k += TILE) {
        const int a_col = tile_k + threadIdx.x;
        const int b_row = tile_k + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : __float2half(0.0f);
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : __float2half(0.0f);

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        const int idx = row * N + col;
        const float old_c = __half2float(C[idx]);
        C[idx] = __float2half_rn(alpha * acc + beta * old_c);
    }
}

}  // namespace

// A, B, and C are device pointers.
extern "C" void solve(const half* A,
                      const half* B,
                      half* C,
                      int M,
                      int N,
                      int K,
                      float alpha,
                      float beta) {
    if (M <= 0 || N <= 0 || K < 0) {
        return;
    }

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}

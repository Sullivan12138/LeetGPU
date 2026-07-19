#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 32;

__device__ int unpack_int4_weight(const unsigned char* __restrict__ w_q,
                                  int n,
                                  int k,
                                  int packed_k) {
    const unsigned char packed = w_q[n * packed_k + (k >> 1)];
    const int nibble = ((k & 1) == 0) ? (packed >> 4) : (packed & 0x0F);
    return nibble - 8;
}

__global__ void int4_fp16_gemm_kernel(const half* __restrict__ x,
                                      const unsigned char* __restrict__ w_q,
                                      const half* __restrict__ scales,
                                      half* __restrict__ y,
                                      int M,
                                      int N,
                                      int K,
                                      int group_size,
                                      int packed_k,
                                      int num_groups) {
    __shared__ half xs[TILE_M][TILE_K + 1];
    __shared__ half ws[TILE_N][TILE_K + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int m = blockIdx.y * TILE_M + ty;
    const int n = blockIdx.x * TILE_N + tx;

    float acc = 0.0f;

    for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {
        for (int linear = tid; linear < TILE_M * TILE_K; linear += TILE_M * TILE_N) {
            const int local_m = linear / TILE_K;
            const int local_k = linear - local_m * TILE_K;
            const int global_m = blockIdx.y * TILE_M + local_m;
            const int global_k = tile_k + local_k;

            xs[local_m][local_k] =
                (global_m < M && global_k < K) ? x[global_m * K + global_k]
                                               : __float2half(0.0f);
        }

        for (int linear = tid; linear < TILE_N * TILE_K; linear += TILE_M * TILE_N) {
            const int local_n = linear / TILE_K;
            const int local_k = linear - local_n * TILE_K;
            const int global_n = blockIdx.x * TILE_N + local_n;
            const int global_k = tile_k + local_k;

            half value = __float2half(0.0f);
            if (global_n < N && global_k < K) {
                const int q = unpack_int4_weight(w_q, global_n, global_k, packed_k);
                const int group = global_k / group_size;
                const float scale = __half2float(scales[global_n * num_groups + group]);
                value = __float2half(static_cast<float>(q) * scale);
            }
            ws[local_n][local_k] = value;
        }

        __syncthreads();

#pragma unroll
        for (int k_inner = 0; k_inner < TILE_K; ++k_inner) {
            acc += __half2float(xs[ty][k_inner]) * __half2float(ws[tx][k_inner]);
        }

        __syncthreads();
    }

    if (m < M && n < N) {
        y[m * N + n] = __float2half_rn(acc);
    }
}

}  // namespace

// x, w_q, scales, and y are device pointers.
// x:      [M, K] half, row-major.
// w_q:    [N, ceil(K / 2)] packed unsigned int4. High nibble stores even k.
// scales: [N, ceil(K / group_size)] half, row-major by output channel then group.
// y:      [M, N] half, row-major.
extern "C" void solve(const half* x,
                      const unsigned char* w_q,
                      const half* scales,
                      half* y,
                      int M,
                      int N,
                      int K,
                      int group_size) {
    if (M <= 0 || N <= 0 || K <= 0 || group_size <= 0) {
        return;
    }

    const int packed_k = (K + 1) / 2;
    const int num_groups = (K + group_size - 1) / group_size;

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    int4_fp16_gemm_kernel<<<grid, block>>>(x, w_q, scales, y, M, N, K, group_size, packed_k, num_groups);
}

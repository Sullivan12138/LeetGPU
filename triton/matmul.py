import torch
import triton
import triton.language as tl



@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):

    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    offsets_n = tl.arange(0, BLOCK_N)
    a_block_ptrs = a + offsets_m[:, None] * stride_am + offsets_n[None, :] * stride_an
    b_block_ptrs = b + offsets_n[:, None] * stride_bn + offsets_k[None, :] * stride_bk

    acc = tl.zeros(
        (BLOCK_M, BLOCK_K),
        dtype=tl.float32,
    )
    for n_start in range(0, N, BLOCK_N):
        a_mask = (offsets_m[:, None] < M) & (n_start + offsets_n[None, :] < N)
        b_mask = (n_start + offsets_n[:, None] < N) & (offsets_k[None, :] < K)

        x = tl.load(a_block_ptrs, mask=a_mask, other=0.0)
        y = tl.load(b_block_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(x, y)
        a_block_ptrs += BLOCK_N * stride_an
        b_block_ptrs += BLOCK_N * stride_bn
    z = acc.to(c.dtype.element_ty)
    c_block_ptrs = c + offsets_m[:, None] * stride_cm + offsets_k[None, :] * stride_ck
    c_mask = (offsets_m[:, None] < M) & (offsets_k[None, :] < K)
    tl.store(c_block_ptrs, z, mask=c_mask)



# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    grid = (
        triton.cdiv(M, 32),
        triton.cdiv(K, 32)
    )
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck, BLOCK_M=32,
        BLOCK_N=32, BLOCK_K=32, num_warps=4
    )

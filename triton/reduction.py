import torch
import triton
import triton.language as tl

@triton.jit
def reduce_local_sum_kernel(input, partial_sums, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N 
    x = tl.load(input + offsets, mask=mask, other=0.0)
    sum_value = tl.sum(x, axis=0)
    tl.store(partial_sums + pid, sum_value)

@triton.jit
def reduce_global_sum_kernel(partial_sums: torch.Tensor, output: torch.Tensor, num_partials: int, BLOCK_SIZE: tl.constexpr):
    acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, num_partials, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_partials
        x = tl.load(partial_sums + offsets, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    tl.store(output, acc)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE=1024
    num_partials = triton.cdiv(N, BLOCK_SIZE)
    grid = (num_partials,)
    partial_sums = torch.empty(num_partials, device=input.device, dtype=torch.float32)
    reduce_local_sum_kernel[grid](input, partial_sums, N, BLOCK_SIZE=BLOCK_SIZE)
    reduce_global_sum_kernel[(1,)](partial_sums, output, num_partials, BLOCK_SIZE=BLOCK_SIZE)

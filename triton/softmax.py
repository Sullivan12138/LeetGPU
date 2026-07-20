import torch
import triton
import triton.language as tl


@triton.jit
def softmax_partial_kernel(input, partial_max, partial_sum, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N 

    x = tl.load(input + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    block_max = tl.max(x, axis=0)

    exp_x = tl.where(mask, tl.exp(x - block_max), 0.0)
    block_sum = tl.sum(exp_x, axis=0)

    tl.store(partial_max + pid, block_max)
    tl.store(partial_sum + pid, block_sum)


@triton.jit
def softmax_global_kernel(input, partial_max, partial_sum, output, num_partials, N, BLOCK_SIZE: tl.constexpr):
    global_max = -float("inf")
    for i in range(0, num_partials, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_partials
        local_max = tl.load(partial_max + offsets, mask=mask, other=-float("inf"))
        global_max = max(global_max, tl.max(local_max, axis=0))

    global_sum = 0.0
    for i in range(0, num_partials, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_partials
        local_sum = tl.load(partial_sum + offsets, mask=mask, other=-float("inf"))
        local_max = tl.load(partial_max + offsets, mask=mask, other=-float("inf"))
        corrected_sum = tl.where(mask, local_sum * tl.exp(local_max-global_max), 0.0)
        global_sum = global_sum + tl.sum(corrected_sum, axis=0)

    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(input + offsets, mask=mask, other=0.0)
        value = tl.exp(x-global_max) / global_sum
        tl.store(output + offsets, value, mask=mask)

    

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    num_partials = triton.cdiv(N, BLOCK_SIZE)
    partial_max = torch.empty((num_partials,), device=input.device, dtype=torch.float32)
    partial_sum = torch.empty((num_partials,), device=input.device, dtype=torch.float32)

    softmax_partial_kernel[(num_partials,)](input, partial_max, partial_sum, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    softmax_global_kernel[(1,)](input, partial_max, partial_sum, output, num_partials, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
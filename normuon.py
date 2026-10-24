"""
Reference from github.com/KellerJordan/modded-nanogpt
---
MIT License

Copyright (c) 2024 Keller Jordan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import defaultdict

import torch
import torch.distributed as dist
import triton
import triton.language as tl


def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]


@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def XXT_kernel(
    A_ptr,
    C_ptr,
    M,
    K,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def XXT(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size
        * triton.cdiv(M, meta["BLOCK_SIZE_M"])
        * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    XXT_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
    )
    return out


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ba_plus_cAA_kernel(
    A_ptr,
    C_ptr,
    M,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from XXT_kernel, but also loads and adds a block of A
    # Performance is slightly slower than XXT_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size
        * triton.cdiv(M, meta["BLOCK_SIZE_M"])
        * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ba_plus_cAA_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
    )
    return out


polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

torch._dynamo.config.cache_size_limit = 12


# Must use dynamic=False or else it's much slower
@torch.compile(dynamic=False, fullgraph=True)
def polar_express(G: torch.Tensor):
    """
    Polar Express Sign Method: https://arxiv.org/pdf/2505.16932
    by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.
    """
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the iterations
    for a, b, c in polar_express_coeffs:
        XXT(X, out=A)  # A = X @ X.mT
        ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X
        X, C = C, X  # Swap references to avoid unnecessary copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class NorMuon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).

    Differences from standard Muon:
    - Newton-Shulz is replaced with Polar Express for the orthogonalization step
    - NorMuon adds a low-rank variance estimator similar to Adafactor.
    - small 1D parameters handled here instead of in Adam
    - Cautious weight decay, a gated version of decoupled weight decay
    - Custom distributed sizing:
    The model stores all attn and mlp weights in the same shape, and then updates the view as
    needed on the forward pass. This enables attn and mlp weights to be contained within the same
    dist.reduce_scatter_tensor() call. The model architecture has been customized to enable
    (n_attn_layers+n_mlp_layers*2)%8==0 for batching across 8 GPUs with zero padding on mlp and attn.
    The scheduling is:
        1. reduce scatter smear_gate (1 param 7 padding params)
        2. reduce scatter attn_gate (10 params 6 padding params)
        3. reduce scatter attn/mlp round 1 (10 attn params 6 mlp params)
        4. reduce scatter attn/mlp round 2 (16 mlp params)
        5. wait on step 1, then compute update of 1 and schedule all gather
        6. wait on step 2, then compute update of 2 and schedule all gather
        7. wait on step 3, then compute update of 3 and schedule all gather
            GPUs receive [2 ATTN, 2 ATTN, 2 ATTN, 2 ATTN, 2 ATTN, 2 MLP, 2 MLP, 2 MLP]
            GPUs that receive params of type attn reshape before computing update
        8. wait on 4, then compute update of 4 and schedule all gather
        9. wait for each all gather to complete and update params
    Empirically, leading with small params provides an additional 0.2s improvement.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        beta2=0.95,
        use_polar_express: bool = True,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            beta2=beta2,
            use_polar_express=use_polar_express,
        )
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        param_groups = self.generate_standard_param_groups(params)
        super().__init__(param_groups, defaults)

    def reset(self):
        # expose a reset for clearing buffers
        for group in self.param_groups:
            group["momentum_buffer"].zero_()
            group["second_momentum_buffer"].zero_()

    def generate_standard_param_groups(self, params):
        """
        Use this method if running on less than 8 GPU or experimenting with additional attn or mlp modules.
        Creates one param group per module.
        """
        groups = defaultdict(list)
        for param in params:
            groups[param.label].append(param)

        param_groups = []
        for module_name, group_params in groups.items():
            chunk_size = (len(group_params) + self.world_size - 1) // self.world_size
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))

        return param_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, @vagrawal, and @varunneal.
        rank = dist.get_rank() if dist.is_initialized() else 0
        all_gather_infos = []
        # Second pass: wait for gradients, compute updates for the local shard of parameters,
        # and launch all async all_gather operations.
        for group in self.param_groups:
            params = group["params"]
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size

            start_idx = rank * chunk_size
            module_idx = start_idx if start_idx < len(params) else 0

            # num params for this rank
            num_params = min(chunk_size, max(0, len(params) - start_idx))

            grad_chunk = torch.empty(
                (chunk_size, *params[0].shape),
                dtype=params[0].dtype,
                device=params[0].device,
            )
            for i, p in enumerate(params[module_idx : module_idx + num_params]):
                grad_chunk[i].copy_(p.grad, non_blocking=True)
            if num_params < chunk_size:
                grad_chunk[num_params:].zero_()

            if "momentum_buffer" not in group:
                group["momentum_buffer"] = torch.zeros_like(grad_chunk[:num_params])
            momentum_buffer = group["momentum_buffer"]
            # Apply momentum update to the persistent momentum buffer in-place
            momentum_buffer.lerp_(grad_chunk[:num_params], 1 - group["momentum"])
            updated_grads = grad_chunk[:num_params].lerp_(
                momentum_buffer, group["momentum"]
            )

            grad_shape = updated_grads.shape
            # reshapes
            for label, N in [
                ("attn.qkv", 3),
                ("mlp.w12", 2),
                ("adaLN_modulation2", 2),
                ("adaLN_modulation4", 4),
                ("adaLN_modulation6", 6),
            ]:
                if params[module_idx].label == label:
                    # Reshape attn params from [hdim, dim*4] to [4,hdim,dim]
                    for p in params[module_idx : module_idx + num_params]:
                        assert p.label == label
                    updated_grads = updated_grads.view(
                        N * grad_shape[0], grad_shape[1] // N, grad_shape[2]
                    )
            ref_param = params[module_idx]
            param_shape = ref_param.shape

            if "second_momentum_buffer" not in group:
                group["second_momentum_buffer"] = (
                    torch.zeros_like(updated_grads[..., :, :1])
                    if param_shape[-2] >= param_shape[-1]
                    else torch.zeros_like(updated_grads[..., :1, :])
                )
            second_momentum_buffer = group["second_momentum_buffer"]

            if "param_lr" not in group:
                group["param_lr"] = max(
                    1.0, param_shape[-2] / param_shape[-1]
                ) ** 0.5 * ref_param.new_tensor(
                    [
                        getattr(param, "lr_mul", 1.0)
                        for param in params[module_idx : module_idx + num_params]
                    ]
                ).view(-1, 1, 1)

                group["param_wd"] = ref_param.new_tensor(
                    [
                        getattr(param, "wd_mul", 1.0)
                        for param in params[module_idx : module_idx + num_params]
                    ]
                ).view(-1, 1, 1)

            # Determine LR and WR
            eff_lr = group["lr"] * group["param_lr"]
            eff_wd = group["lr"] * group["weight_decay"] * group["param_wd"]

            # Compute zeropower for the entire chunk in a single, batched call.
            if num_params == 0:
                v_chunk = updated_grads
            elif group["use_polar_express"]:
                v_chunk = polar_express(updated_grads)
            else:
                v_chunk = zeropower_via_newtonschulz5(updated_grads)

            # NorMuon: second_momentum_buffer tracks squared magnitude of gradients along one dim (https://arxiv.org/pdf/2510.05491)
            v_norm = v_chunk.norm(dim=(-2, -1), keepdim=True)
            v_mean = v_chunk.square().mean(
                dim=-1 if param_shape[-2] >= param_shape[-1] else -2, keepdim=True
            )
            second_momentum_buffer.lerp_(
                v_mean.to(dtype=ref_param.dtype), 1 - group["beta2"]
            )
            step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
            v_chunk.mul_(step_size)
            v_norm_new = v_chunk.norm(dim=(-2, -1), keepdim=True)
            v_chunk.mul_(v_norm / v_norm_new.clamp_min_(1e-10))

            v_chunk = v_chunk.reshape(grad_shape)

            updated_params = torch.empty_like(grad_chunk)
            param_chunk = (
                torch.stack(params[module_idx : module_idx + num_params])
                if num_params > 0
                else torch.zeros_like(v_chunk)
            )

            # "Cautious" weight decay (https://arxiv.org/abs/2510.12402)
            mask = (v_chunk * param_chunk) >= 0
            v_chunk.addcmul_(param_chunk, (eff_wd * mask).to(ref_param.dtype))

            param_chunk.addcmul_(v_chunk, -eff_lr)

            updated_params[:num_params].copy_(param_chunk)
            if num_params < chunk_size:
                updated_params[num_params:].zero_()

            stacked_params = torch.empty(
                (padded_num_params, *param_shape),
                dtype=updated_params.dtype,
                device=updated_params.device,
            )

            if dist.is_initialized():
                gather_future = dist.all_gather_into_tensor(
                    stacked_params, updated_params, async_op=True
                ).get_future()
            else:
                gather_future = None
                stacked_params = updated_params

            all_gather_infos.append(
                {
                    "gather_future": gather_future,
                    "stacked_params": stacked_params,
                    "orig_params": params,
                }
            )

        # Final pass: wait for all_gather to complete and copy results back into original parameter tensors.
        for info in all_gather_infos:
            if (future := info["gather_future"]) is not None:
                future.wait()
            stacked_params = info["stacked_params"]
            orig_params = info["orig_params"]

            unstacked_params = torch.unbind(stacked_params)
            for i, p in enumerate(orig_params):
                p.copy_(unstacked_params[i], non_blocking=True)

        return loss

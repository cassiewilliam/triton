import pytest
import torch

import triton
import triton.language as tl
import triton.tlx.language as tlx
from triton.tools.tensor_descriptor import TensorDescriptor
import os

DEVICE = triton.runtime.driver.active.get_active_torch_device()

TRIPLET_AUTOTUNE = os.getenv("TRIPLET_AUTOTUNE", "0") == "1"

def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_SIZE_KV = nargs["BLOCK_SIZE_KV"]
    HEAD_DIM = nargs["HEAD_DIM"]
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    nargs["desc_k1"].block_shape = [BLOCK_SIZE_KV, HEAD_DIM]
    nargs["desc_k2"].block_shape = [BLOCK_SIZE_KV, HEAD_DIM]
    nargs["desc_v1"].block_shape = [BLOCK_SIZE_KV, HEAD_DIM]
    nargs["desc_v2"].block_shape = [BLOCK_SIZE_KV, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]

def get_configs():
    return [
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_SIZE_KV": BLOCK_SIZE_KV,
                "NUM_BUFFERS": num_buffers,
                "NUM_MMA_WARPS": num_mma_warps,
                "NUM_MMA_GROUPS": num_mma_groups,
            },
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook,
        )
        for BLOCK_SIZE_KV in [64]
        for num_buffers in [2]
        for num_mma_warps in [4]
        for num_mma_groups in [1]
    ]

# Without TMA. Kernel currently used in prod.
@triton.autotune(
    configs=get_configs(),
    key=["HEAD_DIM", "w1", "w2", "seq_len"],
)
@triton.jit
def _triplet_tlx_fwd_kernel(
    desc_q,  # [b, s, k, h]
    desc_k1,  # [b, s, 1, h]
    desc_k2,  # [b, s, 1, h]
    desc_v1,  # [b, s, 1, h]
    desc_v2,  # [b, s, 1, h]
    desc_o,  # [b, s, k, h]
    M_ptr,  # [b, k, s]
    bs,
    seq_len,
    num_heads,
    w1: tl.constexpr,
    w2: tl.constexpr,
    q_stride_b,
    q_stride_s,
    q_stride_k,
    q_stride_h,
    k1_stride_b,
    k1_stride_s,
    k1_stride_k,
    k1_stride_h,
    k2_stride_b,
    k2_stride_s,
    k2_stride_k,
    k2_stride_h,
    v1_stride_b,
    v1_stride_s,
    v1_stride_k,
    v1_stride_h,
    v2_stride_b,
    v2_stride_s,
    v2_stride_k,
    v2_stride_h,
    out_stride_b,
    out_stride_s,
    out_stride_k,
    out_stride_h,
    m_stride_b,
    m_stride_k,
    m_stride_s,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,  #
    NUM_MMA_WARPS: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
):
    # allocate SMEM buffers
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_Q // NUM_MMA_GROUPS
    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS)
    k2_tiles = tlx.local_alloc((BLOCK_SIZE_KV, HEAD_DIM), tlx.dtype_of(desc_k2), NUM_BUFFERS)
    v2_tiles = tlx.local_alloc((BLOCK_SIZE_KV, HEAD_DIM), tlx.dtype_of(desc_v2), NUM_BUFFERS)

    k1_tiles = tlx.local_alloc((1, HEAD_DIM), tlx.dtype_of(desc_k1), w1)
    v1_tiles = tlx.local_alloc((1, HEAD_DIM), tlx.dtype_of(desc_v1), w1)

    # allocate barriers
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=1)

    # k1_tiles, v1_tiles will be loaded from GMEM to SMEM once
    k1_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    v1_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    k2_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS)
    k2_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    v2_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS)
    v2_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)


    with tlx.async_tasks():
        # producer group
        with tlx.async_task("default"):
            # initialize offsets
            q_idx = tl.program_id(0)
            offs_b = tl.program_id(1)

            kv2_start = tl.maximum(0, q_idx - w2 + 1)
            kv2_end = tl.minimum(seq_len, q_idx + 1)

            kv1_idx_start = tl.maximum(0, q_idx - w1 + 1)
            kv1_idx_end = tl.minimum(seq_len, q_idx + 1)
            num_of_kv1_trips = kv1_idx_end - kv1_idx_start
            kv2_offset_y = offs_b * k2_stride_b + kv2_start * k2_stride_s
            # TODO: check the boundary cases
            k1o_offset = offs_b * k1_stride_b + kv1_idx_start * k1_stride_s
            v1o_offset = offs_b * v1_stride_b + kv1_idx_start * v1_stride_s

            # load q: it will stay in SRAM throughout
            for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                q_full = tlx.local_view(q_fulls, cid)
                tlx.barrier_expect_bytes(q_full, 2 * BLOCK_M_SPLIT * HEAD_DIM)
                q_tile = tlx.local_view(q_tiles, cid)
                qo_offset_ysplit = offs_b * q_stride_b + q_idx * q_stride_s + cid * BLOCK_M_SPLIT
                tlx.async_descriptor_load(desc_q, q_tile, [qo_offset_ysplit, 0], q_full)


            k1_full= tlx.local_view(k1_fulls, 0)
            tlx.barrier_expect_bytes(k1_full, 2 * HEAD_DIM * w1)
            k1_tile = tlx.local_view(k1_tiles, 0)
            tlx.async_descriptor_load(desc_k1, k1_tile, [k1o_offset, 0], k1_full)

            v1_full = tlx.local_view(v1_fulls, 0)
            tlx.barrier_expect_bytes(v1_full, 2 * HEAD_DIM * w1)
            v1_tile = tlx.local_view(v1_tiles, 0)
            tlx.async_descriptor_load(desc_v1, v1_tile, [v1o_offset, 0], v1_full)

            # loop over loading k, v
            kv_phase = 0
            acc_cnt = 0

            for _ in tl.range(kv2_start, kv2_end, BLOCK_SIZE_KV):
                buf_id = acc_cnt % NUM_BUFFERS
                # buffers in a row share the same phase
                kv_phase = kv_phase ^ (buf_id == 0)

                # wait for the K buffer to be released by the consumer
                k_empty = tlx.local_view(k2_empties, buf_id)
                tlx.barrier_wait(k_empty, kv_phase)
                # load K
                k2_full = tlx.local_view(k2_fulls, buf_id)
                k2_tile = tlx.local_view(k2_tiles, buf_id)
                tlx.barrier_expect_bytes(k2_full, 2 * BLOCK_SIZE_KV * HEAD_DIM)  # bfloat16
                tlx.async_descriptor_load(desc_k2, k2_tile, [kv2_offset_y, 0], k2_full)

                # wait for the V buffer to be released by the consumer
                v_empty = tlx.local_view(v2_empties, buf_id)
                tlx.barrier_wait(v_empty, kv_phase)
                # load V
                v2_full = tlx.local_view(v2_fulls, buf_id)
                v2_tile = tlx.local_view(v2_tiles, buf_id)
                tlx.barrier_expect_bytes(v2_full, 2 * BLOCK_SIZE_KV * HEAD_DIM)  # bfloat16
                tlx.async_descriptor_load(desc_v2, v2_tile, [kv2_offset_y, 0], v2_full)

                kv2_offset_y += BLOCK_SIZE_KV
                acc_cnt += 1

        # consumer group
        with tlx.async_task(num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS, registers=232, replicate=NUM_MMA_GROUPS):
            # prepare offsets
            q_idx = tl.program_id(0)
            offs_b = tl.program_id(1)

            kv2_start = tl.maximum(0, q_idx - w2 + 1)
            kv2_end = tl.minimum(seq_len, q_idx + 1)

            kv1_idx_start = tl.maximum(0, q_idx - w1 + 1)
            kv1_idx_end = tl.minimum(seq_len, q_idx + 1)
            num_of_kv1_trips = kv1_idx_end - kv1_idx_start

            # kv2_offset_y = offs_b * k2_stride_b + kv2_start * k2_stride_s
            # TODO: check the boundary cases
            # k1o_offset = offs_b * k1_stride_b + kv1_idx_start * k1_stride_s
            # v1o_offset = offs_b * v1_stride_b + kv1_idx_start * v1_stride_s


            # initialize pointer to m and l
            m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
            acc = tl.zeros([BLOCK_M_SPLIT, HEAD_DIM], dtype=tl.float32)

            # load scales
            softmax_scale = tl.cast(SM_SCALE, tlx.dtype_of(desc_q))

            cid = tlx.async_task_replica_id()

            # wait for Q tile
            q_full = tlx.local_view(q_fulls, cid)
            tlx.barrier_wait(q_full, 0)
            q_tile = tlx.local_view(q_tiles, cid)

            # wait for K1 tile
            k1_full = tlx.local_view(k1_fulls, 0)
            tlx.barrier_wait(k1_full, 0)

            # consumer group 0 and consumer group 1 use the same k1 tile
            k1_tile = tlx.local_view(k1_tiles, 0)
            q_tile_rmem = tlx.local_load(q_tile)
            k1_tile_rmem = tlx.local_load(k1_tile)

            # NOTE: we should store qk1_tile in smem after the compute to re-use it in the loop
            qk1_tile_rmem = q_tile_rmem * k1_tile_rmem * softmax_scale

            k2_phase = 0
            v2_phase = 1
            k2_buf_id = 0
            v2_buf_id = 0

            # ===== Section 0: pre-compute QK1@K2 and online softmax for the first iteration =====
            k2_full = tlx.local_view(k2_fulls, k2_buf_id)
            tlx.barrier_wait(k2_full, k2_phase)
            k2_tile = tlx.local_view(k2_tiles, k2_buf_id)
            k2_tile = tlx.local_trans(k2_tile)

            # compute qk[0]
            qk = tlx.async_dot(qk1_tile_rmem, k2_tile)
            qk = tlx.async_dot_wait(0, qk)

            k2_empty = tlx.local_view(k2_empties, k2_buf_id)
            tlx.barrier_wait(k2_empty, 1)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)

            l_ij = tl.sum(p, 1)
            l_i = l_i * alpha + l_ij

            m_i = m_ij
            acc = acc * alpha[:, None]
            acc_cnt = 1
            # ===== Section 0: END =====

            # NOTE: make sure all v1 tilese are loaded before entering the loop
            v1_full = tlx.local_view(v1_fulls, 0)
            tlx.barrier_wait(v1_full, 0)

            for kv1_idx in tl.range(0, num_of_kv1_trips):
                k1_tile = tlx.local_view(k1_tiles, kv1_idx)
                v1_tile = tlx.local_view(v1_tiles, kv1_idx)

                q_tile_rmem = tlx.local_load(q_tile)
                k1_tile_rmem = tlx.local_load(k1_tile)
                v1_tile_rmem = tlx.local_load(v1_tile)

                # NOTE: here we should re-use the qk1_tile from the previous iteration
                qk1_tile_rmem = q_tile_rmem * k1_tile_rmem * softmax_scale

                # loop over k, v and update accumulator
                for _ in tl.range(kv2_start + BLOCK_SIZE_KV, kv2_end, BLOCK_SIZE_KV):
                    # ===== Section 1: compute QK1@K2 for the current iteration =====
                    k2_buf_id = acc_cnt % NUM_BUFFERS
                    k2_phase = k2_phase ^ (k2_buf_id == 0)

                    # wait for the K buffer to be populated by the producer
                    k2_full = tlx.local_view(k2_fulls, k2_buf_id)
                    tlx.barrier_wait(k2_full, k2_phase)
                    k2_tile = tlx.local_view(k2_tiles, k2_buf_id)

                    # compute qk for the current iteration
                    k2_tile = tlx.local_trans(k2_tile) # [HEAD_DIM, BLOCK_SIZE_KV]
                    qk = tlx.async_dot(qk1_tile_rmem, k2_tile, out_dtype=tl.float32)
                    # ===== Secontion 1: END =====


                    # ===== Section 2: compute P@V1V2 from the previous iteration =====
                    # wait for the previous V buffer to be populated by the producer
                    v2_buf_id = (acc_cnt - 1) % NUM_BUFFERS
                    v2_phase = v2_phase ^ (v2_buf_id == 0)
                    v2_full = tlx.local_view(v2_fulls, v2_buf_id)
                    tlx.barrier_wait(v2_full, v2_phase)
                    v2_tile = tlx.local_view(v2_tiles, v2_buf_id) # [BLOCK_SIZE_KV, HEAD_DIM]
                    v2_tile_rmem = tlx.local_load(v2_tile)

                    v12_tile_rmem = v1_tile_rmem * v2_tile_rmem  # [BLOCK_SIZE_KV, HEAD_DIM]
                    tlx.local_store(v2_tile, v12_tile_rmem) # [BLOCK_SIZE_KV, HEAD_DIM]

                    p = p.to(tlx.dtype_of(desc_q)) # [BLOCK_M_SPLIT, BLOCK_SIZE_KV]
                    acc = tlx.async_dot(p, v2_tile, acc) # [BLOCK_M_SPLIT, HEAD_DIM]
                    # ===== Section 2: END =====

                    # ===== Section 3: online softmax for the current iteration =====
                    # wait for the previous qk1 k2 [Section 1] mma to finish
                    qk = tlx.async_dot_wait(1, qk)
                    k2_empty = tlx.local_view(k2_empties, k2_buf_id)
                    tlx.barrier_wait(k2_empty, 1)

                    m_ij = tl.maximum(m_i, tl.max(qk, 1))
                    p = tl.math.exp2(qk - m_ij[:, None])
                    alpha = tl.math.exp2(m_i - m_ij)

                    l_ij = tl.sum(p, 1)
                    l_i = l_i * alpha + l_ij

                    m_i = m_ij
                    # ===== Section 3: END =====

                    # ===== Section 4: update acc for the current iteration =====
                    # wait for the previous p v12 mma [Section 2] to finish
                    acc = tlx.async_dot_wait(0, acc)
                    v2_empty = tlx.local_view(v2_empties, v2_buf_id)
                    tlx.barrier_wait(v2_empty, 0)
                    acc = acc * alpha[:, None]
                    acc_cnt += 1
                    # ===== Section 4: END =====

            # ===== Section 5: compute P@V1V2 for the last iteration =====
            # wait for the last V buffer to be populated by the producer
            v2_buf_id = acc_cnt % NUM_BUFFERS
            v2_phase = v2_phase ^ (v2_buf_id == 0)
            v2_full = tlx.local_view(v2_fulls, v2_buf_id)
            tlx.barrier_wait(v2_full, v2_phase)
            v2_tile = tlx.local_view(v2_tiles, v2_buf_id)
            v2_tile_rmem = tlx.local_load(v2_tile)

            v1_tile = tlx.local_view(v1_tiles, num_of_kv1_trips-1)
            v1_tile_rmem = tlx.local_load(v1_tile)

            v12_tile_rmem = v1_tile_rmem * v2_tile_rmem  # [BLOCK_SIZE_K, HEAD_DIM]
            tlx.local_store(v2_tile, v12_tile_rmem) # [BLOCK_SIZE_K, HEAD_DIM]

            p = p.to(tlx.dtype_of(desc_q))
            acc = tlx.async_dot(p, v2_tile, acc)
            acc = tlx.async_dot_wait(0, acc)
            v2_empty = tlx.local_view(v2_empties, v2_buf_id)
            tlx.barrier_wait(v2_empty, 1)
            # ===== Section 5: END =====

            # ===== Section 6: epilogue =====
            qo_offset_ysplit = offs_b * q_stride_b + q_idx * q_stride_s + cid * BLOCK_M_SPLIT
            desc_o.store([qo_offset_ysplit, 0], acc.to(tlx.dtype_of(desc_o)))
            # ===== Section 6: END =====

def get_tensor_descriptor(
    tensor
):
    bs, seq_len, num_heads, head_dim = tensor.shape

    y_dim = bs * seq_len * num_heads
    dummy_block = [1, 1]

    return TensorDescriptor(
        tensor, shape=[y_dim, head_dim], strides=[head_dim, 1], block_shape=dummy_block
    )

def block_local_triplet_attn_fwd_gqa_pack(
    q, k1, k2, v1, v2, w1, w2,
):
    bs, seq_len, num_heads, head_dim = q.shape
    _, seq_len1, _, _ = k1.shape
    _, seq_len2, _, _ = k2.shape
    assert (
        seq_len == seq_len1 and seq_len1 == seq_len2
    ), "input seq lens must match, sliding window is done within kernel"
    assert w1 > 0 and w2 > 0, "block local windows must be positive"
    output = torch.zeros_like(q, memory_format=torch.contiguous_format).to(
        torch.bfloat16
    )
    m = torch.zeros((bs, num_heads, seq_len), dtype=torch.float32, device=q.device)

    # NOTE: to optimize performance, we always make sure w1 is the smaller window size, when
    # w1 and w2 are not equal.
    if w1 > w2:
        k1, k2 = k2, k1
        v1, v2 = v2, v1
        w1, w2 = w2, w1


    desc_q = get_tensor_descriptor(q)
    desc_k1 = get_tensor_descriptor(k1)
    desc_k2 = get_tensor_descriptor(k2)
    desc_v1 = get_tensor_descriptor(v1)
    desc_v2 = get_tensor_descriptor(v2)
    desc_o = get_tensor_descriptor(output)

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    # INPUT_PRECISION = "ieee"
    # INPUT_PRECISION = "tf32"
    # e^x = 2^(x * log2(e)), so we multiply x by log2(e) to use faster exp2 in kernel.
    sm_scale = 1.44269504  # math.log2(math.exp(1))
    sm_scale *= head_dim**-0.5

    grid = lambda args: (seq_len, bs)

    _triplet_tlx_fwd_kernel[grid](
        desc_q,
        desc_k1,
        desc_k2,
        desc_v1,
        desc_v2,
        desc_o,
        m,
        bs,
        seq_len,
        num_heads,
        w1,
        w2,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k1.stride(0),
        k1.stride(1),
        k1.stride(2),
        k1.stride(3),
        k2.stride(0),
        k2.stride(1),
        k2.stride(2),
        k2.stride(3),
        v1.stride(0),
        v1.stride(1),
        v1.stride(2),
        v1.stride(3),
        v2.stride(0),
        v2.stride(1),
        v2.stride(2),
        v2.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        m.stride(0),
        m.stride(1),
        m.stride(2),
        HEAD_DIM=head_dim,
        SM_SCALE=sm_scale,
        BLOCK_SIZE_Q=num_heads,
    )
    return output, m


def test():
    bs, seq_len, num_heads, head_dim = 1, 1024, 64, 128

    w1 = 512
    w2 = 32

    q = torch.randn((bs, seq_len, num_heads, head_dim), dtype=torch.bfloat16,  device=DEVICE)
    k1 = torch.randn((bs, seq_len, num_heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
    k2 = torch.randn((bs, seq_len, num_heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
    v1 = torch.randn((bs, seq_len, num_heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
    v2 = torch.randn((bs, seq_len, num_heads, head_dim), dtype=torch.bfloat16, device=DEVICE)

    output, _ = block_local_triplet_attn_fwd_gqa_pack(q, k1, k2, v1, v2, w1, w2)

    print(output.shape)

if __name__ == "__main__":
    test()

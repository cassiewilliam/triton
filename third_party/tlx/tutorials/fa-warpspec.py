import pytest
import torch

import triton
import triton.language as tl
import triton.tlx.language as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    HEAD_DIM = nargs["HEAD_DIM"]
    BLOCK_M_SPLIT = BLOCK_M // 2
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'NUM_STAGES': 2}, num_stages=0, num_warps=4,
                      pre_hook=_host_descriptor_pre_hook),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def tlx_attention_fwd(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,
):
    # dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // 2

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    # initialize offsets
    offs_m0 = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
    offs_m1 = start_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
    #offs_n = tl.arange(0, BLOCK_N)

    cst_9 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) - float("inf")
    cst_10 = tl.zeros([BLOCK_M // 2], dtype=tl.float32) + 1.0

    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # allocate buffers for q0, q1
    buffers_q0 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float16, 1)
    buffers_q1 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float16, 1)
    buffer_q0 = tlx.local_view(buffers_q0, 0)
    buffer_q1 = tlx.local_view(buffers_q1, 0)
    barrier_q0 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_q1 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    tlx.async_descriptor_load(desc_q, buffer_q0, [qo_offset_y, 0], barrier_q0)
    tlx.async_descriptor_load(desc_q, buffer_q1, [qo_offset_y + BLOCK_M // 2, 0], barrier_q1)

    # allocate NUM_STAGES buffers for k, v
    buffer_23 = tlx.local_alloc((BLOCK_N, HEAD_DIM), tl.float16, NUM_STAGES)  # k
    buffer_32 = tlx.local_alloc((BLOCK_N, HEAD_DIM), tl.float16, NUM_STAGES)  # v
    buffer_63 = tlx.local_alloc((BLOCK_M // 2, ), tl.float32, 3)  # m_i
    buffer_73 = tlx.local_alloc((BLOCK_M // 2, ), tl.float32, 3)
    buffer_83 = tlx.local_alloc((BLOCK_M // 2, ), tl.float32, 1)
    buffer_84 = tlx.local_alloc((BLOCK_M // 2, ), tl.float32, 1)
    buffer_85 = tlx.local_alloc((BLOCK_M // 2, ), tl.float32, 1)
    buffer_86 = tlx.local_alloc((BLOCK_M // 2, ), tl.float32, 1)

    # allocate tmem
    result = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)
    result_2 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)
    result_3 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)
    result_4 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem)

    # allocate barriers for channels
    barrier_24 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    barrier_27 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    barrier_33 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    barrier_36 = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    barrier_39 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_41 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_43 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_45 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_47 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_49 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_51 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_53 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_55 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_57 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_59 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_61 = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    barrier_64 = tlx.alloc_barriers(num_barriers=3, arrive_count=1)
    barrier_65 = tlx.alloc_barriers(num_barriers=3, arrive_count=1)
    barrier_74 = tlx.alloc_barriers(num_barriers=3, arrive_count=1)
    barrier_75 = tlx.alloc_barriers(num_barriers=3, arrive_count=1)

    # causal = False
    lo, hi = 0, N_CTX
    with tlx.async_tasks():
        with tlx.async_task("default"):  # correction
            # set up arguments
            arg26, arg27, arg28, arg30 = 0, 0, 0, 0
            arg29 = -1
            for start_n in tl.range(lo, hi, BLOCK_N):
                # data slice 0
                # convert from ttng.wait_barrier to tlx.barrier_wait, trace the barrier and the phase
                val_116 = arg26 ^ 1
                val_117 = arg27 + 1
                val_121 = 1 if val_117 == 3 else val_117
                val_120 = arg28 ^ 1 if val_117 == 3 else arg28
                view_1 = tlx.local_view(buffer_63, val_121)
                view_2 = tlx.local_view(barrier_64, val_121)
                view_3 = tlx.local_view(barrier_65, val_121)
                tlx.barrier_wait(view_2, val_120)
                m_i = tlx.local_load(view_1, tlx.storage_kind.smem)  # m_i
                tlx.barrier_arrive(view_3, 1)

                val_126 = arg29 + 1
                val_130 = 1 if val_126 == 3 else val_126
                val_129 = arg30 ^ 1 if val_126 == 3 else arg30
                view_4 = tlx.local_view(buffer_63, val_130)
                view_5 = tlx.local_view(barrier_64, val_130)
                view_6 = tlx.local_view(barrier_65, val_130)
                tlx.barrier_wait(view_5, val_129)
                m_ij = tlx.local_load(view_4)  # m_ij
                tlx.barrier_arrive(view_6, 1)

                alpha = tl.math.exp2(m_i - m_ij)
                view_7 = tlx.local_view(barrier_45, 0)
                tlx.barrier_wait(view_7, arg26)  # acc0
                # subtiling to reduce register pressure when hDim is 128
                view_8 = tlx.local_view(result_2, 0)
                view_9 = tlx.subslice(view_8, 0, HEAD_DIM // 2)  # N = 0
                view_10 = tlx.subslice(view_8, HEAD_DIM // 2, HEAD_DIM // 2)  # N = 64
                result_7 = tlx.local_load(view_9, tlx.storage_kind.tmem)
                val_141 = result_7 * alpha[:, None]
                tlx.local_store(view_9, val_141, tlx.storage_kind.tmem)
                result_8 = tlx.local_load(view_10, tlx.storage_kind.tmem)
                val_142 = result_8 * alpha[:, None]
                tlx.local_store(view_10, val_142, tlx.storage_kind.tmem)
                view_11 = tlx.local_view(barrier_43, 0)
                tlx.barrier_arrive(view_11, 1)  # acc0

                # data slice 1
                view_12 = tlx.local_view(buffer_73, val_121)
                view_13 = tlx.local_view(barrier_74, val_121)
                view_14 = tlx.local_view(barrier_75, val_121)
                tlx.barrier_wait(view_13, val_120)
                m_i1 = tlx.local_load(view_12, tlx.storage_kind.smem)  # m_i
                tlx.barrier_arrive(view_14, 1)

                view_15 = tlx.local_view(buffer_73, val_130)
                view_16 = tlx.local_view(barrier_74, val_130)
                view_17 = tlx.local_view(barrier_75, val_130)
                tlx.barrier_wait(view_16, val_129)
                m_ij1 = tlx.local_load(view_15)  # m_ij
                tlx.barrier_arrive(view_17, 1)

                alpha1 = tl.math.exp2(m_i1 - m_ij1)
                view_18 = tlx.local_view(barrier_57, 0)
                tlx.barrier_wait(view_18, arg26)  # acc1
                # subtiling to reduce register pressure when hDim is 128
                view_19 = tlx.local_view(result, 0)
                view_20 = tlx.subslice(view_19, 0, HEAD_DIM // 2)  # N = 0
                view_21 = tlx.subslice(view_19, 64, HEAD_DIM // 2)  # N = 64
                result_9 = tlx.local_load(view_20, tlx.storage_kind.tmem)
                val_157 = result_9 * alpha1[:, None]
                tlx.local_store(view_20, val_157, tlx.storage_kind.tmem)
                result_10 = tlx.local_load(view_21, tlx.storage_kind.tmem)
                val_158 = result_10 * alpha1[:, None]
                tlx.local_store(view_21, val_158, tlx.storage_kind.tmem)
                view_22 = tlx.local_view(barrier_55, 0)
                tlx.barrier_arrive(view_22, 1)  # acc1

                # update loop variables
                arg26 = val_116
                arg27 = val_121
                arg28 = val_120
                arg29 = val_130
                arg30 = val_129

        with tlx.async_task(num_warps=1):  # gemm
            # dot0_slice0
            view_1 = tlx.local_view(barrier_24, 0)
            view_2 = tlx.local_view(barrier_27, 0)
            view_3 = tlx.local_view(buffer_23, 0)  # k
            view_4 = tlx.local_trans(view_3)
            tlx.barrier_wait(view_2, 0)
            view_5 = tlx.local_view(barrier_41, 0)
            tlx.barrier_wait(view_5, 0)
            view_6 = tlx.local_view(result_3, 0)
            # dot0_slice1: q0 . k
            view_7 = tlx.local_view(barrier_39, 0)
            tlx.async_dot(buffer_q0, view_4, view_6, mBarriers=[view_1, view_7])
            view_8 = tlx.local_view(barrier_53, 0)
            tlx.barrier_wait(view_8, 0)  # has predicate?
            view_9 = tlx.local_view(result_4, 0)
            view_10 = tlx.local_view(barrier_51, 0)
            tlx.async_dot(buffer_q1, view_4, view_9, mBarriers=[view_1, view_10])

            arg65, arg66, arg67 = 0, 0, 0
            for start_n in tl.range(lo, hi, BLOCK_N):
                val_121 = hi - 128
                # dot1_slice0_iter_i
                view_11 = tlx.local_view(barrier_33, arg65)
                view_12 = tlx.local_view(barrier_36, arg65)
                view_13 = tlx.local_view(buffer_32, arg65)
                tlx.barrier_wait(view_12, arg66)
                view_14 = tlx.local_view(barrier_43, 0)
                val_126 = arg67 ^ 1
                tlx.barrier_wait(view_14, val_126)
                tlx.barrier_wait(barrier_49, arg67)
                view_15 = tlx.local_reinterpret(view_6, tl.float16)
                view_16 = tlx.local_view(result_2, 0)
                view_17 = tlx.local_view(barrier_45, 0)
                view_t = tlx.local_view(barrier_47, 0)
                tlx.async_dot(view_15, view_13, view_16, mBarriers=[view_11, view_17, view_t])  #barrier_47])
                # dot0_slice0_iter_i+1
                val_128 = arg65 + 1
                val_131 = 0 if val_128 == 2 else val_128
                val_132 = arg66 ^ 1 if val_128 == 2 else arg66
                view_18 = tlx.local_view(barrier_24, val_131)
                view_19 = tlx.local_view(barrier_27, val_131)
                view_20 = tlx.local_view(buffer_23, val_131)
                view_21 = tlx.local_trans(view_20)
                tlx.barrier_wait(view_19, val_132)  #, pred=start_n < val_121)
                tlx.barrier_wait(view_5, arg67 ^ 1)  #, pred=start_n < val_121)
                tlx.async_dot(buffer_q0, view_21, view_6, mBarriers=[view_18, view_7])  #pred=start_n < val_121
                view_22 = tlx.local_reinterpret(view_9, tl.float16)
                view_23 = tlx.local_view(barrier_55, 0)
                tlx.barrier_wait(view_23, arg67 ^ 1)
                tlx.barrier_wait(barrier_61, arg67)
                # dot1_slice1_iter_i
                view_24 = tlx.local_view(result, 0)
                view_25 = tlx.local_view(barrier_57, 0)
                view_t = tlx.local_view(barrier_59, 0)
                tlx.async_dot(view_22, view_13, view_24, mBarriers=[view_11, view_25, view_t])
                tlx.barrier_wait(view_8, arg67 ^ 1)  #, pred=start_n < val_121)
                # dot0_slice1_iter_i+1
                tlx.async_dot(buffer_q1, view_21, view_9, mBarriers=[view_18, view_10])  #pred=start_n < val_121

                # update arg65/arg66/arg67
                arg65 = val_131
                arg66 = val_132
                arg67 = val_126

        with tlx.async_task(num_warps=1):  # load
            view_1 = tlx.local_view(barrier_24, 0)
            tlx.barrier_wait(view_1, 0)
            view_2 = tlx.local_view(barrier_27, 0)
            tlx.barrier_expect_bytes(view_2, 32768)
            view_3 = tlx.local_view(buffer_23, 0)
            tlx.async_descriptor_load(desc_k, view_3, [offset_y, 0], view_2)

            view_4 = tlx.local_view(barrier_24, 1)
            tlx.barrier_wait(view_4, 0)  #, pred=hi > 128)
            view_5 = tlx.local_view(barrier_27, 1)
            tlx.barrier_expect_bytes(view_5, 32768)
            view_6 = tlx.local_view(buffer_23, 1)
            tlx.async_descriptor_load(desc_k, view_6, [offset_y + 128, 0], view_5)  #, pred=hi > 128)
            # load k_iter_0, k_iter_1
            arg65 = offset_y + 128
            arg66 = 1
            arg67, arg68, arg69 = 0, 0, 0
            arg70 = offset_y
            for start_n in tl.range(lo, hi, BLOCK_N):
                start_n = tl.multiple_of(start_n, BLOCK_N)
                val_126 = hi - 256
                # load v_iter_i
                view_7 = tlx.local_view(barrier_33, arg68)
                tlx.barrier_wait(view_7, arg69)
                view_8 = tlx.local_view(barrier_36, arg68)
                tlx.barrier_expect_bytes(view_8, 32768)
                view_9 = tlx.local_view(buffer_32, arg68)
                tlx.async_descriptor_load(desc_v, view_9, [arg70, 0], view_8)

                # load k_iter_i+2 with predicate
                val_132 = arg68 + 1
                val_135 = 0 if val_132 == 2 else val_132
                val_136 = arg69 ^ 1 if val_132 == 2 else arg69
                val_137 = arg65 + 128
                val_138 = arg66 + 1
                val_141 = 0 if val_138 == 2 else val_138
                val_142 = arg67 ^ 1 if val_138 == 2 else arg67
                view_10 = tlx.local_view(barrier_24, val_141)
                tlx.barrier_wait(view_10, val_142)  #, pred=start_n < val_126)
                view_11 = tlx.local_view(barrier_27, val_141)
                tlx.barrier_expect_bytes(view_11, 32768)
                view_12 = tlx.local_view(buffer_23, val_141)
                tlx.async_descriptor_load(desc_k, view_12, [val_137, 0], view_11)  #, pred=start_n < val_126)

                # update args
                arg70 = arg65
                arg65 = val_137
                arg66 = val_141
                arg67 = val_142
                arg68 = val_135
                arg69 = val_136

        with tlx.async_task(num_warps=4):  # softmax0
            arg65 = cst_10
            arg66 = cst_9
            arg67, arg68, arg69 = 0, 0, 0
            for start_n in tl.range(lo, hi, BLOCK_N):
                view_1 = tlx.local_view(barrier_39, 0)
                tlx.barrier_wait(view_1, arg67)
                view_2 = tlx.local_view(result_3, 0)
                result_14 = tlx.local_load(view_2, tlx.storage_kind.tmem)
                view_3 = tlx.local_view(barrier_41, 0)
                tlx.barrier_arrive(view_3, 1)

                m_ij = tl.maximum(arg66, tl.max(result_14, 1) * qk_scale)

                val_122 = arg68 + 1
                val_125 = arg69 ^ 1 if val_122 == 3 else arg69
                val_126 = 1 if val_122 == 3 else val_122
                view_4 = tlx.local_view(buffer_63, val_126)
                view_5 = tlx.local_view(barrier_64, val_126)
                view_6 = tlx.local_view(barrier_65, val_126)
                tlx.barrier_wait(view_6, val_125)
                tlx.local_store(view_4, m_ij)
                tlx.barrier_arrive(view_5, 1)

                val_133 = result_14 * qk_scale - m_ij[:, None]
                val_134 = tl.math.exp2(val_133)  # p
                val_136 = tl.math.exp2(arg66 - m_ij)  # alpha
                val_137 = tl.sum(val_134, 1)  # l_ij
                p = val_134.to(tl.float16)
                view_7 = tlx.local_view(result_3, 0)
                view_8 = tlx.local_reinterpret(view_7, tl.float16)
                tlx.barrier_wait(barrier_47, arg67)
                tlx.local_store(view_8, p, tlx.storage_kind.tmem)
                tlx.barrier_arrive(barrier_49, 1)
                val_141 = arg65 * val_136 + val_137  # new value for l_i

                arg65 = val_141
                arg66 = m_ij
                arg67 = arg67 ^ 1
                arg68 = val_126
                arg69 = val_125
        tlx.local_store(buffer_84, arg65)
        tlx.local_store(buffer_83, arg66)

        with tlx.async_task(num_warps=4):  # softmax1
            arg65 = cst_10
            arg66 = cst_9
            arg67, arg68, arg69 = 0, 0, 0
            for start_n in tl.range(lo, hi, BLOCK_N):
                view_1 = tlx.local_view(barrier_51, 0)
                tlx.barrier_wait(view_1, arg67)
                view_2 = tlx.local_view(result_4, 0)
                result_14 = tlx.local_load(view_2, tlx.storage_kind.tmem)
                view_3 = tlx.local_view(barrier_53, 0)
                tlx.barrier_arrive(view_3, 1)

                m_ij = tl.maximum(arg66, tl.max(result_14, 1) * qk_scale)

                val_122 = arg68 + 1
                val_125 = arg69 ^ 1 if val_122 == 3 else arg69
                val_126 = 1 if val_122 == 3 else val_122
                view_4 = tlx.local_view(buffer_73, val_126)
                view_5 = tlx.local_view(barrier_74, val_126)
                view_6 = tlx.local_view(barrier_75, val_126)
                tlx.barrier_wait(view_6, val_125)
                tlx.local_store(view_4, m_ij)
                tlx.barrier_arrive(view_5, 1)

                val_133 = result_14 * qk_scale - m_ij[:, None]
                val_134 = tl.math.exp2(val_133)  # p
                val_136 = tl.math.exp2(arg66 - m_ij)  # alpha
                val_137 = tl.sum(val_134, 1)  # l_ij
                p = val_134.to(tl.float16)
                view_7 = tlx.local_view(result_4, 0)
                view_8 = tlx.local_reinterpret(view_7, tl.float16)
                tlx.barrier_wait(barrier_59, arg67)
                tlx.local_store(view_8, p, tlx.storage_kind.tmem)
                tlx.barrier_arrive(barrier_61, 1)
                val_141 = arg65 * val_136 + val_137  # new value for l_i

                arg65 = val_141
                arg66 = m_ij
                arg67 = arg67 ^ 1
                arg68 = val_126
                arg69 = val_125
        tlx.local_store(buffer_86, arg65)
        tlx.local_store(buffer_85, arg66)

    # epilogue
    view_buffer_86 = tlx.local_view(buffer_86, 0)
    val_88 = tlx.local_load(view_buffer_86)
    view_buffer_85 = tlx.local_view(buffer_85, 0)
    val_90 = tlx.local_load(view_buffer_85)
    view_buffer_84 = tlx.local_view(buffer_84, 0)
    val_91 = tlx.local_load(view_buffer_84)
    view_buffer_83 = tlx.local_view(buffer_83, 0)
    val_93 = tlx.local_load(view_buffer_83)

    val_95 = val_93 + tl.math.log2(val_91)
    m_ptrs = M + off_hz * N_CTX + offs_m0
    tl.store(m_ptrs, val_95)
    view_1 = tlx.local_view(result_2, 0)
    result_5 = tlx.local_load(view_1, tlx.storage_kind.tmem)
    val_102 = result_5 / val_91[:, None]  # check layout
    acc = val_102.to(tl.float16)
    #buffer_104 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float16, 1)
    # fence_async_shared
    tlx.async_descriptor_store(desc_o, acc, [qo_offset_y, 0])
    # tma_store_wait

    val_107 = val_90 + tl.math.log2(val_88)
    m_ptrs = M + off_hz * N_CTX + offs_m1
    tl.store(m_ptrs, val_107)
    view_2 = tlx.local_view(result, 0)
    result_6 = tlx.local_load(view_2, tlx.storage_kind.tmem)
    val_111 = result_6 / val_88[:, None]  # check layout
    acc = val_111.to(tl.float16)
    #buffer_113 = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), tl.float16, 1)
    # fence_async_shared
    tlx.async_descriptor_store(desc_o, acc, [qo_offset_y + BLOCK_M_SPLIT, 0])
    # tma_store_wait


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        extra_kern_args = {}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]

        dummy_block = [1, 1]
        desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        if q.dtype == torch.float8_e5m2:
            desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1], block_shape=dummy_block)
        else:
            desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        tlx_attention_fwd[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o


attention = _attention.apply


@pytest.mark.parametrize("Z", [8])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("N_CTX", [1024])
@pytest.mark.parametrize("HEAD_DIM", [128])
@pytest.mark.parametrize("mode", ["fwd"])
@pytest.mark.parametrize("provider", ["triton-fp16"])
def test_op(Z, H, N_CTX, HEAD_DIM, mode, provider, dtype=torch.float16):
    if mode == "bwd":
        pytest.skip("Backward pass not supported.")
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    # reference implementation
    ref_dtype = dtype
    if mode == "fwd" and "fp8" in provider:
        ref_dtype = torch.float32
    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    # M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1)
    p = p.to(ref_dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v).half()
    # triton implementation
    if mode == "fwd" and "fp8" in provider:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    tri_out = attention(q, k, v, sm_scale).half()
    if mode == "fwd":
        atol = 3 if "fp8" in provider else 1e-2
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return
    # compare
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = False
BATCH, N_HEADS, HEAD_DIM = 4, 32, 128
# vary seq length for fixed head and batch=4
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(10, 15)],
        line_arg="provider",
        line_vals=["triton-fp16"] + (["flash"] if HAS_FLASH else []),
        line_names=["Triton [FP16]"] + (["Flash-2"] if HAS_FLASH else []),
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name=f"fused-attention-ws-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "HEAD_DIM": HEAD_DIM,
            "mode": "fwd",
        },
    ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)

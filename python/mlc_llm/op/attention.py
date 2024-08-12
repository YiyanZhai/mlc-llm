"""Operators enabled by external modules."""

import math
import tvm
from tvm import tir
from tvm.script import tir as T
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op

from mlc_llm.support import logging

from mlc_llm.nn.kv_cache import _attention_prefill_ragged

from . import extern as _extern

logger = logging.getLogger(__name__)


WARN_FLASHINFER_GROUP_SIZE = False
WARN_FLASHINFER_HEAD_DIM = False


def attention(  # pylint: disable=invalid-name,too-many-locals,too-many-statements,too-many-arguments
    q: nn.Tensor,
    k: nn.Tensor,
    v: nn.Tensor,
    casual_mask: nn.Tensor,
    attn_score_scaling_factor: float = 1.0,
    qk_dtype: str = None,
) -> nn.Tensor:
    """Attention with casual mask.

    --- Variables ---
    s: sequence length of the current query
    t: total sequence length
    d: head dimension
    h, h_q: number of heads in query
    h_kv: number of heads in key and value
    b: batch size = 1

    --- Shapes ---
    q: [b, s, h_q, d]
    k: [t, h_kv, d]
    v: [t, h_kv, d]
    o: [1, s, hidden = h_q * d]

    --- Computation ---

    .. code-block:: python

        if h_kv != h_q:
            k = k.repeat(h_q // h_kv, axis=1)
            v = v.repeat(h_q // h_kv, axis=1)
        q -> [b, h, s, d]
        k, v -> [b, h, t, d]
        attn = q @ k^T / sqrt(d) * attn_score_scaling_factor  # [b, h, s, t]
        attn = softmax_with_mask(attn, casual_mask, axis=-1)
        o = attn @ v  # [b, h, s, d]
        o -> [b, s, h * d]

    --- Other params ---
    qk_dtype: if set, `matmul(Q, K, out_dtype=qk_dtype)`, (otherwise use `q.dtype` as `out_dtype`).
        For FlashInfer, if "float32", sets `allow_fp16_qk_reduction` to False; otherwise no effectir.
    """
    assert q.ndim == 4 and k.ndim in [3, 4] and v.ndim in [3, 4]
    b, s, h_q, d = q.shape
    t, h_kv, _ = k.shape[-3:]
    group_size = h_q // h_kv

    def _fallback():
        from tvm import relax as rx
        nonlocal q, k, v, qk_dtype
        if h_kv != h_q:
            k = k.repeat(h_q // h_kv, axis=2)
            v = v.repeat(h_q // h_kv, axis=2)
        q = op.reshape(q, [b*s, h_q, d])
        k = op.reshape(k, [b*s, h_q, d])
        v = op.reshape(v, [b*s, h_q, d])
        q_rope_position = op.zeros([b*s], dtype="float16")
        k_rope_pos_offset = op.zeros([1], dtype="float16")
        causal = T.int32(1)
        if casual_mask is None:
            causal = T.int32(0)
        rotary_mode = T.int32(0)
        rope_scale = T.float32(1.0)
        rope_theta = T.float32(0.0)
        attn_score_scaling_factor = T.float32(1.0)
        o = op.empty([b*s, h_q, d], dtype="float32")
        lse = op.empty([b*s, h_q], dtype="float32")
        func = rx.BlockBuilder().current().add_func(_attention_prefill_ragged(h_q, h_q, d, q.dtype, [], tvm.target.Target("cuda")), func_name="batch_prefill_ragged_kv")
        rx.op.call_pure_packed("batch_prefill_ragged_kv", func,
                q._expr, k._expr, v._expr,
                q_rope_position._expr, k_rope_pos_offset._expr,
                causal, rotary_mode,
                rope_scale, rope_theta,
                attn_score_scaling_factor,
                o._expr, lse._expr,
                sinfo_args=rx.ObjectStructInfo())
        # result = op.tensor_ir_op(
        #     _attention_prefill_ragged(h_q, h_q, d, q.dtype, [], tvm.target.Target("cuda")),
        #     name_hint="batch_prefill_ragged_kv",
        #     args=[
        #         q, k, v,
        #         q_rope_position, k_rope_pos_offset,
        #         causal, rotary_mode,
        #         rope_scale, rope_theta,
        #         attn_score_scaling_factor
        #     ],
        #     out = [nn.Tensor.placeholder((b*s), (h_q), (d), "float32"), nn.Tensor.placeholder((T.int64(b*s), T.int64(h_q)), "float32")],
        #     # out=[nn.Tensor.placeholder(q.shape, q.dtype), nn.Tensor.placeholder((q.shape[0], q.shape[1]), "float32")],
        # )
        return op.wrap_nested(o, "attn_output")

    # FlashInfer Implementation
    if (
        _extern.get_store().flashinfer
        and attn_score_scaling_factor == 1.0
        and q.dtype == "float16"
        and k.dtype == "float16"
        and v.dtype == "float16"
    ):
        if group_size not in [1, 4, 6, 8]:
            global WARN_FLASHINFER_GROUP_SIZE  # pylint: disable=global-statement
            if not WARN_FLASHINFER_GROUP_SIZE:
                WARN_FLASHINFER_GROUP_SIZE = True
                logger.warning(
                    "FlashInfer only supports group size in [1, 4, 6, 8], but got %d. Skip and "
                    "fallback to default implementation.",
                    group_size,
                )
            return _fallback()
        if d not in [128]:
            global WARN_FLASHINFER_HEAD_DIM  # pylint: disable=global-statement
            if not WARN_FLASHINFER_HEAD_DIM:
                WARN_FLASHINFER_HEAD_DIM = True
                logger.warning(
                    "FlashInfer only supports head_dim in [128], but got %d. Skip and fallback to "
                    "default implementation.",
                    d,
                )
            return _fallback()
        rope_theta = 0.0
        rope_scale = 1.0
        qkv_layout = 0  # "NHD", N for seq_len, H for num_heads, D for head_dim
        rotary_mode = 0  # "kNone"
        casual = 1  # True
        fp16_qk = 1  # True
        if qk_dtype == "float32":
            fp16_qk = 0  # False

        # 32MB scratchpad
        scratch = op.empty([8192 * 1024], dtype="float32")  # pylint: disable=no-member

        def _decode():
            return op.extern(
                name="flashinfer.single_decode",
                args=[
                    q,
                    k,
                    v,
                    scratch,
                    qkv_layout,
                    rotary_mode,
                    rope_scale,
                    rope_theta,
                ],
                out=nn.Tensor.placeholder((b, s, h_q * d), dtype="float16"),
            )

        def _prefill():
            return op.extern(
                name="flashinfer.single_prefill",
                args=[
                    q,
                    k,
                    v,
                    scratch,
                    casual,
                    qkv_layout,
                    rotary_mode,
                    fp16_qk,
                    rope_scale,
                    rope_theta,
                ],
                out=nn.Tensor.placeholder((b, s, h_q * d), dtype="float16"),
            )

        if isinstance(s, int) and s == 1:
            func = "decode"
        else:
            func = "prefill"
        return {
            "decode": _decode,
            "prefill": _prefill,
        }[func]()

    # Fallback Implementation
    return _fallback()

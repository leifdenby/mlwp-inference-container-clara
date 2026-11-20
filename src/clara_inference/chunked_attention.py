"""
Memory-saving self-attention for CLARA's U-Nets, for larger domains.

neural-lam's `UNetBlock` (`neural_lam.models.edm_networks_2`) computes its
self-attention weights for all query/key pairs at once (`softmax(QᵀK)`, in
float32). On the DANRA domain the largest attention (the grid U-Net's
bottleneck, 198 x 148 = 29,304 positions) needs ~3.4 GB per head; on a 4x
larger domain it needs 16x that (52 GB for 376 x 316) and doesn't fit.

`patch_unet_attention` makes the blocks compute the attention for chunks of
queries in turn. Each query's softmax is over all keys either way, so the
result is the same, with the same operations and precision per chunk; only
the peak memory drops (chunk x keys instead of queries x keys).

Run neural-lam with it as
`python -m clara_inference.chunked_attention <train_model arguments>`, with
the chunk size in `CLARA_ATTENTION_CHUNK` (unset or 0: no patch).
"""
import os
import runpy
import sys

import numpy as np
import torch
from loguru import logger

ENV_CHUNK = "CLARA_ATTENTION_CHUNK"


def attend_in_chunks(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk: int
) -> torch.Tensor:
    """
    Compute softmax(QᵀK / sqrt(C)) V for chunks of queries, as
    `UNetBlock.forward` does for all queries at once.

    Parameters
    ----------
    q, k, v : torch.Tensor
        (N, C, L) queries, keys and values (N = batch x heads).
    chunk : int
        Queries per chunk.

    Returns
    -------
    torch.Tensor
        (N, C, L) attention output.
    """
    k_scaled = (k / np.sqrt(k.shape[1])).to(torch.float32)
    q32 = q.to(torch.float32)
    parts = []
    for start in range(0, q.shape[2], chunk):
        w = (
            torch.einsum(
                "ncq,nck->nqk", q32[:, :, start : start + chunk], k_scaled
            )
            .softmax(dim=2)
            .to(q.dtype)
        )
        parts.append(torch.einsum("nqk,nck->ncq", w, v))
        del w
    return torch.cat(parts, dim=2)


def patch_unet_attention(chunk: int) -> None:
    """
    Make neural-lam's `UNetBlock` compute its self-attention in query chunks.

    Parameters
    ----------
    chunk : int
        Queries per chunk.
    """
    from neural_lam.models import edm_networks_2

    block = edm_networks_2.UNetBlock
    if getattr(block, "_clara_attention_chunk", None):
        block._clara_attention_chunk = chunk
        return
    forward = block.forward

    def forward_chunked(self, x, emb=None):
        num_heads = self.num_heads
        if not num_heads:
            return forward(self, x, emb)
        # everything before the attention, as in UNetBlock.forward
        self.num_heads = 0
        try:
            x = forward(self, x, emb)
        finally:
            self.num_heads = num_heads
        q, k, v = (
            self.qkv(self.norm2(x))
            .reshape(x.shape[0] * num_heads, x.shape[1] // num_heads, 3, -1)
            .unbind(2)
        )
        a = attend_in_chunks(q, k, v, block._clara_attention_chunk)
        x = self.proj(a.reshape(*x.shape)).add_(x)
        return x * self.skip_scale

    block.forward = forward_chunked
    block._clara_attention_chunk = chunk
    logger.info(f"U-Net self-attention in chunks of {chunk} queries")


def main() -> None:
    chunk = int(os.environ.get(ENV_CHUNK, "0") or 0)
    if chunk > 0:
        patch_unet_attention(chunk)
    sys.argv = ["neural_lam.train_model"] + sys.argv[1:]
    runpy.run_module("neural_lam.train_model", run_name="__main__")


if __name__ == "__main__":
    main()

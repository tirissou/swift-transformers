import pdb
from coremltools.converters.mil.frontend.milproto.load import TranscriptionContext
from requests import head
import torch
from torch.functional import Tensor
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple
import coremltools


class SlidingCache(Cache):
    def __init__(
        self,
        shape: Tuple[int, ...],
        # device="cpu",
        # max_length=2048,
        dtype=torch.half,
    ) -> None:
        """KV cache of shape (#layers, #heads, head_dim, 1, seq len)."""
        # note THIS DOES NOT deal with batch size
        super().__init__()
        self.keyCache: Tensor = torch.zeros(shape, dtype=dtype)
        self.valueCache: Tensor = torch.zeros(shape, dtype=dtype)

    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        head_idx: int,
        cache_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update kv cache at [layer_idx, head_idx, :, :, :]"""
        self.keyCache[layer_idx, head_idx, :, :, :, :] = torch.concat((self.keyCache[layer_idx, head_idx, :, :, :, k_state.shape[-1]:]), k_state)

    #
    # def get_seq_length(self, _: int | None = 0) -> Tensor:
    #     """Get the sequence length of the cache as 0-dim Tensor"""
    #     return torch.min(torch.max(self.cacheSequenceLength), self.max_length)
    #


import torch
from torch.functional import Tensor
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple

class SliceUpdateKeyValueCache(Cache):
    def __init__(
        self,
        shape: Tuple[int, ...],
        device="cpu",
        max_length=2048,
        dtype=torch.half,
    ) -> None:
        """KV cache of shape (#layers, #heads, cacheLength, 1, head_dim)."""
        # note THIS DOES NOT deal with batch size
        super().__init__()

        self.max_length = Tensor([max_length]).to(torch.long)
        self.cacheLength: Tensor
        self.cacheStartIndex: Tensor
        self.keyCache: Tensor
        self.valueCache: Tensor
        
        self.register_buffer('cacheLength', torch.zeros([32, 32, 1], dtype=torch.long))
        self.register_buffer('cacheStartIndex', torch.zeros([32, 32, 1], dtype=torch.long))
        self.register_buffer('keyCache', torch.full(shape, torch.nan, dtype=dtype))
        self.register_buffer('valueCache', torch.full(shape, torch.nan, dtype=dtype))

    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        head_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update kv cache at [layer_idx, head_idx, :, :, :]"""
        # TODO redo logic using only singular 

        new_seq_len = k_state.shape[-1]
        cacheLength = self.cacheLength[layer_idx, head_idx]
        start = self.cacheStartIndex[layer_idx, head_idx]

        # TODO handle case where initial sequence is larger than max size
        # first update the cache from start -> start + new_seq_len
        self.keyCache[layer_idx, head_idx, :, :, start:start+new_seq_len] = k_state
        self.valueCache[layer_idx, head_idx, :, :, start:start+new_seq_len] = v_state
        # then update start
        cacheLength += new_seq_len
        self.cacheStartIndex = torch.where(cacheLength >= self.max_length, cacheLength % self.max_length, 0)
        # then slice up the cache to "fix" the rotation
        k_cache = torch.concat(
            (
                self.keyCache[layer_idx, head_idx, :, :, start:cacheLength],
                self.keyCache[layer_idx, head_idx, :, :, :start]
            ), dim=-1
        )
        v_cache = torch.concat(
            (
                self.valueCache[layer_idx, head_idx, :, :, start:cacheLength],
                self.valueCache[layer_idx, head_idx, :, :, :start]
            ), dim=-1
        )

        return k_cache, v_cache

    def get_seq_length(self, _: int | None = 0) -> Tensor:
        """Get the sequence length of the cache as 0-dim Tensor"""
        return torch.min(torch.max(self.cacheLength), self.max_length)


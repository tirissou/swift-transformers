
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
        self.cacheSequenceLength: Tensor
        self.keyCache: Tensor
        self.valueCache: Tensor
        
        self.register_buffer('cacheSequenceLength', torch.zeros([1], dtype=torch.long))
        self.register_buffer('keyCache', torch.full(shape, torch.nan, dtype=dtype))
        self.register_buffer('valueCache', torch.full(shape, torch.nan, dtype=dtype))

    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        head_idx: int,
        cache_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update kv cache at [layer_idx, head_idx, :, :, :]"""
        # TODO redo logic using only singular 

        # maximum of self.max_length
        # TODO unsqueeze batch dimension if enabling batch
        k_state = k_state[..., -self.max_length:]
        v_state = v_state[..., -self.max_length:]

        # calculate where to start reading from AFTER updating
        highest_position = torch.max(cache_position)
        start = torch.where(
            highest_position > self.max_length, 
            highest_position % self.max_length, 
            torch.tensor([0], dtype=torch.long)
        )
        cache_position = cache_position[..., -self.max_length:] % self.max_length

        self.keyCache[layer_idx, head_idx, :, :, cache_position] = k_state
        self.valueCache[layer_idx, head_idx, :, :, cache_position] = v_state

        k_cache = torch.concat(
            (
                self.keyCache[layer_idx, head_idx, :, :, start:highest_position],
                self.keyCache[layer_idx, head_idx, :, :, :start]
            ), dim=-1
        )
        v_cache = torch.concat(
            (
                self.valueCache[layer_idx, head_idx, :, :, start:highest_position],
                self.valueCache[layer_idx, head_idx, :, :, :start]
            ), dim=-1
        )

        self.cacheSequenceLength = highest_position

        return k_cache, v_cache

    def get_seq_length(self, _: int | None = 0) -> Tensor:
        """Get the sequence length of the cache as 0-dim Tensor"""
        return torch.min(torch.max(self.cacheSequenceLength), self.max_length)


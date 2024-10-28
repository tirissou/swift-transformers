import pdb
from coremltools.converters.mil.frontend.milproto.load import TranscriptionContext
import torch
from torch.functional import Tensor
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple
import coremltools


class SliceUpdateKeyValueCache(Cache):
    def __init__(
        self,
        shape: Tuple[int, ...],
        device="cpu",
        max_length=2048,
        dtype=torch.half,
    ) -> None:
        """KV cache of shape (#layers, #heads, head_dim, 1, seq len)."""
        # note THIS DOES NOT deal with batch size
        super().__init__()

        self.max_length = torch.tensor([max_length], dtype=torch.long)
        self.positions = torch.arange(max_length)
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
        # TODO redo logic using only singular 

        # maximum of self.max_length
        # TODO unsqueeze batch dimension if enabling batch
        # k_state = k_state[..., -self.max_length:]
        # v_state = v_state[..., -self.max_length:]

        # calculate where to start reading from AFTER updating
        highest_position = torch.max(cache_position) + 1
        start = torch.where(
            highest_position > self.max_length, 
            highest_position % self.max_length, 
            torch.tensor([0], dtype=torch.long)
        )
        highest_position = torch.min(
            highest_position, self.max_length
        )

        # cache_position = cache_position[..., -self.max_length:] % self.max_length
        cache_position = cache_position % self.max_length

        # Does removing the below two lines let me go further in execution?
        # Could add a batch dimension?
        # Could try using some index_copy_ magic
        # self.keyCache[layer_idx, head_idx].index_copy_(-1, cache_position, k_state.squeeze(0))
        # self.valueCache[layer_idx, head_idx].index_copy_(-1, cache_position, v_state.squeeze(0))
        assert self.valueCache[layer_idx, head_idx, :, :, cache_position].shape == v_state.squeeze(0).shape
        assert self.valueCache.dtype == v_state.dtype
        self.valueCache[layer_idx, head_idx, :, :, cache_position] = v_state.squeeze(0)
        self.keyCache[layer_idx, head_idx, :, :, cache_position] = k_state.squeeze(0)

        # edge case appears when start=0 as the second last dim ends up size 0
        read_indices = self.positions[:highest_position] + start
        k_cache = self.keyCache[layer_idx, head_idx, :, :, read_indices]
        v_cache = self.valueCache[layer_idx, head_idx, :, :, read_indices]



        # k_cache = torch.where(
        #     start == 0,
        #     torch.concat(
        #         (
        #             self.keyCache[layer_idx, head_idx, :, :, start:highest_position],
        #             self.keyCache[layer_idx, head_idx, :, :, :start]
        #         ), dim=-1
        #     )
        # )
        # v_cache = torch.where(
        #     start == 0,
        #     self.valueCache[layer_idx, head_idx, :, :, :highest_position],
        #     torch.concat(
        #         (
        #             self.valueCache[layer_idx, head_idx, :, :, start:highest_position],
        #             self.valueCache[layer_idx, head_idx, :, :, :start]
        #         ), dim=-1
        #     )
        # )
        # # self.cacheSequenceLength = highest_position

        return k_cache, v_cache
    #
    # def get_seq_length(self, _: int | None = 0) -> Tensor:
    #     """Get the sequence length of the cache as 0-dim Tensor"""
    #     return torch.min(torch.max(self.cacheSequenceLength), self.max_length)
    #

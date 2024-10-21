from typing import List, Optional, Tuple
from coremltools import RangeDim
import torch
from transformers.models.mistral.modeling_mistral import (
    MISTRAL_ATTENTION_CLASSES,
    MistralConfig,
    MistralForCausalLM,
)

from .attention import SliceUpdateMistralAttention
from ..cache import SliceUpdateKeyValueCache
from ..helpers import map_weights_linear_to_conv2d

class StatefulMistralForCausalLM(torch.nn.Module):
    def __init__(self, model_path: str, max_context_size: int = 2048, batch_size: int = 1) -> None:
        super().__init__()

        # Custom attention implementation for stateful slice update key/value cache, override
        # "sdpa" to compliance with transformers.modeling_utils._autoset_attn_implementation
        MISTRAL_ATTENTION_CLASSES["sdpa"] = SliceUpdateMistralAttention
        self.model = MistralForCausalLM.from_pretrained(model_path)
        self._register_load_state_dict_pre_hook(map_weights_linear_to_conv2d)

        # Register KV cache buffers to be recognized as Core ML states
        config: MistralConfig = self.model.config
        self.kv_cache_shape: Tuple[int | RangeDim, ...] = (
            config.num_hidden_layers,
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
            1,
            max_context_size,
        )
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape)
        # self.register_buffer("keyCache", self.kv_cache.key_cache)
        # self.register_buffer("valueCache", self.kv_cache.value_cache)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Compute past seen tokens used for updating key/value cache slices
        return self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits



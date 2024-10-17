from itertools import cycle
import logging
import math
import os
import pdb
import warnings
from typing import List, Optional, Tuple

from torch.functional import Tensor
from coreml_tools import *


import coremltools as ct
from coremltools.converters import RangeDim
import numpy as np
import torch
from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import (
    MISTRAL_ATTENTION_CLASSES,
    MistralAttention,
    MistralConfig,
    MistralForCausalLM,
    rotate_half,
)

# warnings.filterwarnings("ignore")
logging.getLogger("coremltools").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
MODEL_ID: str = "mistralai/Mistral-7B-Instruct-v0.3"
METADATA_TOKENIZER: str = "co.huggingface.exporters.name"

def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
    """ Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights
    """
    for k in state_dict:
        is_internal_proj = all(substr in k for substr in ['lin', '.weight'])
        is_output_proj = all(substr in k
                             for substr in ['classifier', '.weight'])
        if is_internal_proj or is_output_proj:
            if len(state_dict[k].shape) == 2:
                state_dict[k] = state_dict[k][:, :, None, None]

def apply_rotary_pos_emb_head(q, k, cos, sin):
    """Rotates hidden head dimensions"""
    # TODO VERIFY
    # IDEAL q, k: (batch, seq, 1, head_dim)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class SliceUpdateKeyValueCache(Cache):
    def __init__(
        self,
        shape: Tuple[int, ...],
        device="cpu",
        dtype=torch.float32,
    ) -> None:
        """KV cache of shape (#layers, batch_size, #kv_heads, context_size, head_dim)."""
        super().__init__()
        self.past_seen_tokens: int = 0
        self.register_buffer('key_cache', torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer('value_cache', torch.zeros(shape, dtype=dtype, device=device))

    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        head_idx: int,
        slice_indices: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update key/value cache tensors for slice [slice_indices[0], slice_indices[1]).
        Return slice of key/value cache tensors from [0, slice_indices[1]).
        """
        if len(slice_indices) != 2:
            raise ValueError(f"Expect tuple of integers [start, end), got {slice_indices=}.")
        begin, end = slice_indices
        self.key_cache[layer_idx, :, : k_state.shape[1], begin:end, :] = k_state
        self.value_cache[layer_idx, :, : v_state.shape[1], begin:end, :] = v_state
        k_cache: torch.Tensor = self.key_cache[layer_idx, :, :, :end, :]
        v_cache: torch.Tensor = self.value_cache[layer_idx, :, :, :end, :]
        return k_cache, v_cache

    def get_seq_length(self, _: int | None = 0) -> int:
        """Get the sequence length of the cache."""
        return self.past_seen_tokens


class SliceUpdateMistralAttention(MistralAttention):
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        self.scale = 1 / math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(config.attention_dropout)


    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[SliceUpdateKeyValueCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor | None, ...]:

        #######################################################################
        # NOTES
        #
        # Both query+keys are multiplied elementwise w/ cos+sin
        # -> head dimension must be last dimension
        #
        # cos+sin should be calculated once per forward invocation
        #
        # No need to rerotate keys as RoPE preserves ABSOLUTE positional information.
        # Just slide the window.
        #
        #
        #
        #######################################################################

        bsz, q_len, _ = hidden_states.size()

        # **Convert from BSC to BCIS (Batch, Channels, Height, Width).**
        # Original shape (bsz, seq len, hidden size)
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)  # Shape: (bsz, hidden_size, seq_len, 1)
        # IDEAL (bsz, hidden_size, 1, seq_len)

        # **Linear projections using Conv2D layers.**
        q: Tensor = self.q_proj(hidden_states)
        k: Tensor  = self.k_proj(hidden_states)
        v: Tensor  = self.v_proj(hidden_states)
        # all (1, hid_dim, 1, seq)

        mh_q = q.split(self.head_dim, dim=1)  # List of tensors: (bsz, head_dim, seq_len, 1)
        mh_v = v.split(self.head_dim, dim=1)
        # (batch, 128, 1, seq)

        # Transpose k and split along the last dimension
        mh_k = k.transpose(1, 3).split(self.head_dim, dim=3)  
        # n_kv_heads * (batch, seq, 1, 128)

        # Calculate rotation for this point in the sequence (not per head)
        cos, sin = self.rotary_emb(mh_v[0], position_ids)
        # (1, 2, 128)
        cos = cos.unsqueeze(2)
        sin = cos.unsqueeze(2)
        # TODO VERIFY
        # (1, 2, 1, 128)

        # Apply rotary embeddings to each head individually
        mh_q_rot = []
        mh_k_rot = []
        # TODO increase 
        for qi, ki in zip(mh_q, cycle(mh_k)):
            # original qi shape: (1, 128, 2, 1)
            # original ki shape: (1, 2, 1, 128)
            # Reshape qi and ki to (bsz, head_dim, seq_len)
            qi = qi.permute(0, 2, 3, 1) # expect shape (1, 2, 1, 128)

            # Apply rotary position embeddings
            qi_rot, ki_rot = apply_rotary_pos_emb_head(qi, ki, cos, sin)
            # both shape (bsz, seq_len, 1, head_dim)

            # Reshape back to original shapes
            # qi_rot = qi_rot.unsqueeze(-1)  # (bsz, head_dim, seq_len, 1)
            # ki_rot = ki_rot.permute(0, 2, 1).unsqueeze(2)  # (bsz, seq_len, 1, head_dim)

            mh_q_rot.append(qi_rot)
            mh_k_rot.append(ki_rot)

        # **Update Key/Value Cache Per Head**
        updated_mh_k = []
        updated_mh_v = []

        end_step = attention_mask.shape[-1]
        seq_start = end_step - q_len
        seq_end = end_step


        for head_idx, (ki, vi) in enumerate(zip(mh_k_rot, cycle(mh_v))):
            # ki: (bsz, seq_len, 1, head_dim)
            # vi: (bsz, head_dim, seq_len, 1)

            # Reshape vi to (bsz, head_dim, seq_len)
            vi: Tensor
            vi = vi.squeeze(-1)  # (bsz, head_dim, seq_len)


            # Update cache per head
            start = head_idx * self.head_dim
            k_cache, v_cache = past_key_value.update(
                ki.squeeze(2),  # (bsz, seq_len, head_dim)
                vi.permute(0, 2, 1),  # (bsz, seq_len, head_dim)
                self.layer_idx,
                torch.LongTensor([start, start + self.head_dim])
            )

            # Append updated caches
            updated_mh_k.append(k_cache)  # (bsz, total_seq_len, head_dim)
            updated_mh_v.append(v_cache)  # (bsz, total_seq_len, head_dim)

        # **Compute Attention per Head using `einsum`**
        attn_outputs = []
        pdb.set_trace()
        for qi, ki, vi in zip(mh_q_rot, updated_mh_k, updated_mh_v):
            # qi: (bsz, head_dim, seq_len, 1)
            # ki: (bsz, total_seq_len, head_dim)
            # vi: (bsz, total_seq_len, head_dim)

            # Reshape ki and vi to match einsum expectations
            ki = ki.unsqueeze(2)  # (bsz, total_seq_len, 1, head_dim)
            vi = vi.permute(0, 2, 1).unsqueeze(-1)  # (bsz, head_dim, 1, total_seq_len)

            # **Compute attention scores using einsum**
            attn_scores = torch.einsum('bchq,bkhc->bkhq', qi, ki) * self.scale  # (bsz, total_seq_len, 1, seq_len)

            # **Apply attention mask**
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask[:, :, None, :]

            # **Compute attention probabilities**
            attn_probs = torch.softmax(attn_scores, dim=1)
            attn_probs = self.dropout(attn_probs)

            # **Compute attention output using einsum**
            attn_output = torch.einsum('bkhq,bchk->bchq', attn_probs, vi)  # (bsz, head_dim, 1, seq_len)
            attn_outputs.append(attn_output)

        # **Concatenate heads and project output**
        attn_output = torch.cat(attn_outputs, dim=1)  # (bsz, num_heads * head_dim, 1, seq_len)

        # Reshape for output projection
        attn_output = attn_output.permute(0, 1, 3, 2)  # (bsz, num_heads * head_dim, seq_len, 1)
        attn_output = self.o_proj(attn_output)  # (bsz, hidden_size, seq_len, 1)

        # Reshape back to original format
        attn_output = attn_output.squeeze(-1).permute(0, 2, 1)  # (bsz, seq_len, hidden_size)

        return attn_output, None, None


class StatefulMistralForCausalLM(torch.nn.Module):
    def __init__(self, model_path: str, max_context_size: int = 2048, batch_size: int = 1) -> None:
        super().__init__()

        # Custom attention implementation for stateful slice update key/value cache, override
        # "sdpa" to compliance with transformers.modeling_utils._autoset_attn_implementation
        MISTRAL_ATTENTION_CLASSES["sdpa"] = SliceUpdateMistralAttention
        self.model = MistralForCausalLM.from_pretrained(model_path)
        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

        # Register KV cache buffers to be recognized as Core ML states
        config: MistralConfig = self.model.config
        self.kv_cache_shape: Tuple[int | RangeDim, ...] = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_context_size,
            config.hidden_size // config.num_attention_heads,
        )
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape)
        self.register_buffer("keyCache", self.kv_cache.key_cache)
        self.register_buffer("valueCache", self.kv_cache.value_cache)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Compute past seen tokens used for updating key/value cache slices
        self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]
        return self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits


def export() -> None:
    # Construct model from transformers and trace to TorchScript
    max_context_size: int = 2048
    torch_model = StatefulMistralForCausalLM(MODEL_ID, max_context_size=max_context_size)
    torch_model.eval()
    input_ids: torch.Tensor = torch.zeros((1, 2), dtype=torch.int32)
    causal_mask: torch.Tensor = torch.zeros((1, 1, 2, 5), dtype=torch.float32)
    linear_to_conv2d(torch_model)
    
    import pdb
    pdb.set_trace()
    traced_model = torch.jit.trace(torch_model, [input_ids, causal_mask])

    # Convert traced TorchScript to Core ML format
    print("Converting from TorchScript to Core ML format")
    query_length = ct.RangeDim(lower_bound=1, upper_bound=max_context_size, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=max_context_size, default=1)
    inputs: List[ct.TensorType] = [
        ct.TensorType(shape=(1, query_length), dtype=np.int32, name="inputIds"),
        ct.TensorType(
            shape=(1, 1, query_length, end_step_dim),
            dtype=np.float16,
            name="causalMask",
        ),
    ]
    outputs: List[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]
    states: List[ct.StateType] = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=torch_model.kv_cache_shape, dtype=np.float16),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=torch_model.kv_cache_shape, dtype=np.float16),
            name="valueCache",
        ),
    ]

    # Convert model with FP16 precision
    print("Converting with FP16 precision")
    mlmodel_fp16: ct.MLModel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.transform.FP16ComputePrecision
    )

    # Block-wise quantize model weights to int4
    print("Quantizing to int4")
    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=32,
    )
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(mlmodel_fp16, config=config)
    mlmodel_int4._spec.description.metadata.userDefined.update({METADATA_TOKENIZER: MODEL_ID})
    mlmodel_int4.save("StatefulMistral7BInstructInt4.mlpackage")


if __name__ == "__main__":
    export()

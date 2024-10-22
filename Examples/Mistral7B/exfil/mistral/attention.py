from itertools import cycle
import math
import pdb
from torch import nn
from typing import List, Optional, Tuple
from torch.functional import Tensor
import torch
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralConfig,
)
from ..cache import SliceUpdateKeyValueCache
from ..helpers import apply_rotary_pos_emb_head

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
        cache_position: torch.Tensor,
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

        # bsz, q_len, #heads x head_size
        hidden_states = hidden_states.transpose(1, 2)

        # **Convert from BSC to BCIS (Batch, Channels, Height, Width).**
        # If (bsz, seq, #heads x head_size) -> (bsz, seq, 1, #heads x head_size)
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.unsqueeze(2) 
        # IDEAL (bsz, #heads x head_size, 1, seq_len)

        # **Linear projections using Conv2D layers.**
        q: Tensor = self.q_proj(hidden_states)
        k: Tensor  = self.k_proj(hidden_states)
        v: Tensor  = self.v_proj(hidden_states)
        # all (1, hid_dim, 1, seq)

        mh_q = q.split(self.head_dim, dim=1)  # List of tensors: (bsz, head_dim, seq_len, 1)
        mh_v = v.split(self.head_dim, dim=1)
        mh_k = k.split(self.head_dim, dim=1)
        # n_kv_heads * (batch, 128, 1, seq)

        # Calculate rotation for this point in the sequence (not per head)
        cos, sin = self.rotary_emb(mh_v[0], position_ids)
        # (1, 2, 128)
        cos = cos.transpose(1, 2).unsqueeze(2)
        sin = sin.transpose(1, 2).unsqueeze(2)
        # confirmed (1, 128, 1, 2)

        # Apply rotary embeddings to each head individually
        mh_q_rot = []
        mh_k_rot = []
        # TODO increase 
        for qi, ki in zip(mh_q, cycle(mh_k)):
            # Apply rotary position embeddings
            qi_rot, ki_rot = apply_rotary_pos_emb_head(qi, ki, cos, sin)

            mh_q_rot.append(qi_rot.to(torch.half))
            mh_k_rot.append(ki_rot.to(torch.half))


        # **Update Key/Value Cache Per Head**
        updated_mh_k = []
        updated_mh_v = []
        for head_idx, (ki, vi) in enumerate(zip(mh_k_rot, cycle(mh_v))):
            # ki AND vi: (bsz, head_dim, 1, seq_len)

            # Update cache per attention head (not KV head)
            assert type(self.layer_idx) is int
            start = head_idx * self.head_dim
            k_cache, v_cache = past_key_value.update(
                ki,
                vi,
                self.layer_idx,
                head_idx,
                cache_position,
            )

            # Append updated caches
            updated_mh_k.append(k_cache)
            updated_mh_v.append(v_cache)  

        # **Compute Attention per Head using `einsum`**
        attn_outputs = []
        for qi, ki, vi in zip(mh_q_rot, updated_mh_k, updated_mh_v):
            # all (bsz, head_dim, 1, seq)
            qi = qi.to(torch.half)
            ki = ki.unsqueeze(0).transpose(1, 3)
            vi = vi.unsqueeze(0)

            # **Compute attention scores using einsum**
            pdb.set_trace()
            attn_scores = torch.einsum('bhcq,bkch->bcqk', qi, ki) * self.scale  # (bsz, total_seq_len, 1, seq_len)

            # **Apply attention mask**
            # attn_scores = attn_scores + attention_mask[:, :, None, :]

            # **Compute attention probabilities**
            attn_probs = torch.softmax(attn_scores, dim=1)
            attn_probs = self.dropout(attn_probs)

            # **Compute attention output using einsum**
            attn_output = torch.einsum('bhcq,bkch->bcqk', attn_probs, vi)  # (bsz, head_dim, 1, seq_len)
            attn_outputs.append(attn_output)

        # **Concatenate heads and project output**
        attn_output = torch.cat(attn_outputs, dim=1)  # (bsz, num_heads * head_dim, 1, seq_len)

        # Reshape for output projection
        attn_output = attn_output.permute(0, 1, 3, 2)  # (bsz, num_heads * head_dim, seq_len, 1)
        attn_output = self.o_proj(attn_output)  # (bsz, hidden_size, seq_len, 1)

        # Reshape back to original format
        attn_output = attn_output.squeeze(-1).permute(0, 2, 1)  # (bsz, seq_len, hidden_size)

        return attn_output, None, None



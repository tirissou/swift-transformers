import torch
from torch.functional import Tensor
from torch import nn

# Function to replace Linear layers with Conv2d layers
def replace_linear_with_conv2d(module):
    for name, layer in module.named_children():
        # If the layer is a Linear layer, replace it with Conv2d
        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
            out_features = layer.out_features

            # Replace the Linear layer with Conv2d
            conv2d_layer = nn.Conv2d(
                in_channels=in_features,   # Input channels same as in_features
                out_channels=out_features, # Output channels same as out_features
                kernel_size=1,             # 1x1 kernel to mimic fully connected behavior
                bias=(layer.bias is not None)
            )
            
            # Copy the weights from Linear to Conv2d
            conv2d_layer.weight.data = layer.weight.data.view(out_features, in_features, 1, 1)
            if layer.bias is not None:
                conv2d_layer.bias.data = layer.bias.data

            # Set the new Conv2d layer
            setattr(module, name, conv2d_layer)
        
        # Recursively apply the transformation to child modules
        else:
            replace_linear_with_conv2d(layer)

def map_weights_linear_to_conv2d(state_dict):
    """ Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights
    """
    for k in state_dict:
        is_internal_proj = all(substr in k for substr in ['lin', '.weight'])
        is_output_proj = all(substr in k
                             for substr in ['classifier', '.weight'])
        if is_internal_proj or is_output_proj:
            if len(state_dict[k].shape) == 2:
                state_dict[k] = state_dict[k][:, :, None, None]

def rotate_half(x: Tensor, dim=-1):
    """Rotates half the hidden dims of the input."""
    half_size = x.shape[dim] // 2
    xs = x.split(half_size, dim=dim)
    x1 = x[..., : x.shape[dim] // 2]
    x2 = x[..., x.shape[dim] // 2 :]
    return torch.cat((-xs[1], xs[0]), dim=dim)

def apply_rotary_pos_emb_head(q, k, cos, sin):
    """Rotates hidden head dimensions"""
    # TODO VERIFY
    # IDEAL q, k: (batch, seq, 1, head_dim)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    # assert q_rot.dtype == torch.half # FAILS
    return q_rot, k_rot



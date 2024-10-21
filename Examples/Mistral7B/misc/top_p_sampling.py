import torch
import coremltools as ct
import numpy as np

# NOTES
# May need to implement torch.searchsorted as coreml does not have an impl yet?

class TopPSampling(torch.nn.Module):
    def __init__(self, p=0.9):
        super(TopPSampling, self).__init__()
        self.p = p

    def forward(self, logits: torch.Tensor):
        # Step 1: Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(logits, dim=-1)

        # Step 2: Sort the probabilities and their corresponding token indices
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

        # Step 3: Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Step 4: Find the cutoff index where cumulative probability exceeds p
        cutoff_index = torch.searchsorted(cumulative_probs, torch.tensor(self.p))

        # Step 5: Keep only tokens up to the cutoff index
        sorted_probs = sorted_probs[:cutoff_index + 1]
        sorted_indices = sorted_indices[:cutoff_index + 1]

        return sorted_probs, sorted_indices  # Return both the probabilities and the indices

# Create a sample logits tensor (e.g., vocab size of 50257)
vocab_size = 8
max_top_p = 2

logits = torch.randn(vocab_size)

# Initialize the top-p sampling model
top_p_model = TopPSampling(p=0.9)

# Trace the model using TorchScript
traced_model = torch.jit.trace(top_p_model, logits)

# Export the traced model to CoreML using coremltools
# Define the input and output types for CoreML
top_p_len = ct.RangeDim(lower_bound=1, upper_bound=max_top_p, default=1)
input_spec = ct.TensorType(name="logits", shape=logits.shape, dtype=np.half)
output_spec_probs = ct.TensorType(name="sorted_probs", dtype=np.half)
output_spec_indices = ct.TensorType(name="sorted_indices", dtype=np.long)

# Convert to CoreML model
coreml_model = ct.convert(
    traced_model,
    inputs=[input_spec],
    outputs=[output_spec_probs, output_spec_indices],
    minimum_deployment_target=ct.target.macOS15,
)

# Save the CoreML model to disk
coreml_model.save("top_p_sampling.mlmodel")
print("CoreML model saved.")

import torch
import coremltools as ct
from typing import List
import numpy as np
from torch._C import dtype

class DynamicSizeArange(torch.nn.Module):
    """
    A basic usage of cond based on dynamic shape predicate.
    """

    def __init__(self, size=8):
        super().__init__()
        self.register_buffer('counter', torch.tensor([0], dtype=torch.half))
        self.register_buffer('cache', torch.zeros(1024, 1, 2048, 1, 128, dtype=torch.half))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # for i in range(1024):
            # self.cache[i] = 
        # self.cache[:, :, :1, :, :] = x
        # self.cache += 0
        start = self.counter.int()
        end = start + 1
        self.counter += 1
        # cache = self.cache[0]
        self.cache[0, :, start:end, :, :] = x
        return self.cache[0, :, 0:end, :, :]






# test_input = torch.arange(4, dtype=torch.half)
test_input = torch.randn(1, 1, 1, 128, dtype=torch.half)

buffer_size = 8
torch_model = DynamicSizeArange(buffer_size)
query_length = ct.RangeDim(lower_bound=1, upper_bound=32, default=1)
inputs: List[ct.TensorType] = [
    ct.TensorType(shape=ct.Shape(shape=(1, 1, 1, 128)), name="input_array", dtype=np.half)
]

states: List[ct.StateType] = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(1024, 1, 2048, 1, 128), dtype=np.half),
        name="cache",
    ),
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(1, ), dtype=np.half),
        name="counter",
    ),
]
outputs: List[ct.TensorType] = [ct.TensorType(dtype=np.half, name="logits")]
# torch_model.eval()
model = torch.jit.trace(torch_model, test_input)
mlmodel_fp16 = ct.convert(
        model,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        # skip_model_load=True,
        compute_units=ct.ComputeUnit.ALL,     # WORKS
        # compute_units=ct.ComputeUnit.CPU_AND_GPU,
        # compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
        debug=True,
        compute_precision=ct.transform.FP16ComputePrecision(),
    )
mlmodel_fp16.save("realistic_basic_state_seq_dim1.mlpackage")

state = mlmodel_fp16.make_state()

print(mlmodel_fp16.predict({"input_array": test_input.numpy()}, state))



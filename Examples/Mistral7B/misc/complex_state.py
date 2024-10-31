import torch
import coremltools as ct
from typing import List
import numpy as np

class DynamicSizeArange(torch.nn.Module):
    """
    A basic usage of cond based on dynamic shape predicate.
    """

    def __init__(self, size=8):
        super().__init__()
        self.register_buffer('data', torch.zeros((4, 4, size), dtype=torch.half))
        self.register_buffer('counter', torch.zeros((1,), dtype=torch.half))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.counter.add_(x.shape[-1])
        return self.counter.to(torch.int) % x.shape[-1]
        # return self.data[..., (self.counter.to(torch.int) % x.shape[-1]):]

test_input = torch.arange(20, dtype=torch.half)

buffer_size = 8
torch_model = DynamicSizeArange(buffer_size)
query_length = ct.RangeDim(lower_bound=1, upper_bound=100, default=1)
inputs: List[ct.TensorType] = [
    ct.TensorType(shape=ct.Shape(shape=(query_length,)), name="input_array", dtype=np.half)
]

states: List[ct.StateType] = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(4, 4, 8,), dtype=np.half),
        name="data",
    ),
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(1,), dtype=np.half),
        name="counter",
    ),
]
outputs: List[ct.TensorType] = [ct.TensorType(dtype=np.half, name="logits")]
torch_model.eval()
model = torch.jit.trace(torch_model, test_input)
mlmodel_fp16 = ct.convert(
        model,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.iOS18,
        # skip_model_load=True,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
        debug=True,
        # compute_precision=ct.transform.FP16ComputePrecision(),
    )
mlmodel_fp16.save("basicstate.mlpackage")

state = mlmodel_fp16.make_state()

print(mlmodel_fp16.predict({"input_array": test_input.numpy()}, state))



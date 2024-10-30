import torch
import coremltools as ct
from typing import List
import numpy as np

class DynamicSizeArange(torch.nn.Module):
    """
    A basic usage of cond based on dynamic shape predicate.
    """

    def __init__(self):
        super().__init__()
        # self.register_buffer('counter', torch.zeros((4, 10), dtype=torch.half))
        self.register_buffer('counter', torch.zeros((1,), dtype=torch.half))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # index = self.counter.to(torch.int32) % x.shape[0]
        # rval = torch.concat((x[index:], x[:index]))
        self.counter.add_(1)
        return self.counter

test_input = torch.arange(20, dtype=torch.half)

torch_model = DynamicSizeArange()
query_length = ct.RangeDim(lower_bound=1, upper_bound=30, default=1)
inputs: List[ct.TensorType] = [
    ct.TensorType(shape=ct.Shape(shape=(query_length,)), name="input_array", dtype=np.half)
]

states: List[ct.StateType] = [
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

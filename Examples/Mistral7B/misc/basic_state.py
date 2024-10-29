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
        self.register_buffer('counter', torch.zeros([1], dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        index = self.counter % x.shape[0]
        if x.dim() == 1:
            rval = torch.concat((x[self.counter:], x[:self.counter]))
        else:
            rval = x[self.counter:]

        self.counter += 1
        return rval

torch_model = DynamicSizeArange()
query_length = ct.RangeDim(lower_bound=1, upper_bound=30, default=1)
inputs: List[ct.TensorType] = [
    ct.TensorType(shape=ct.Shape(shape=(query_length,)), name="input_array")
]

states: List[ct.StateType] = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(1,), dtype=np.float16),
        name="counter",
    ),
]
outputs: List[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]
model = torch.jit.trace(torch_model, torch.zeros(5))
model.eval()
mlmodel_fp16 = ct.convert(
        model,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        debug=True
        # compute_precision=ct.transform.FP16ComputePrecision(),
    )
mlmodel_fp16.save("basicstate.mlpackage")

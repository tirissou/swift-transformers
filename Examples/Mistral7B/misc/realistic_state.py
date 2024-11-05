import torch
import coremltools as ct
from typing import List
import numpy as np
from torch._C import dtype
from torch.functional import Tensor

class DynamicSizeArange(torch.nn.Module):
    """
    A basic usage of cond based on dynamic shape predicate.
    """

    def __init__(self, size=8, max_sequence_length=2048, buffer = 1, hidden_size=128):
        super().__init__()
        self.size = size
        self.max_sequence_length = max_sequence_length
        self.register_buffer('counter', torch.tensor([0], dtype=torch.half))
        self.register_buffer('cache', torch.zeros(1024, self.max_sequence_length, buffer, 1, hidden_size, dtype=torch.half))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # index: Tensor = torch.remainder(self.counter, self.max_sequence_length)
        # print(f"{index=}")
        self.cache[0, (torch.remainder(self.counter, self.max_sequence_length))[0].int()] = x
        self.counter += 1
        start = torch.where(self.counter > self.max_sequence_length, (torch.remainder(self.counter + 1, self.max_sequence_length)), 0)
        index = torch.max(torch.concat((start, torch.tensor([1])))).int()
        # print(f"{start=}")
        # print(f"{index=}")
        # return self.cache + 0

        rval = torch.where(
            start == 0,
            self.cache[0],
            torch.concat((self.cache[0, index:], self.cache[0, :index],), dim=0)
        )
        # 
        rval = rval.permute(1, 0, 2, 3)

        rval = torch.einsum(
            'abcd,adce->abce', rval, rval.transpose(1,3)
        )

        rval = torch.einsum(
            'abcd,adce->abce', rval, rval.transpose(1,3)
        )



        return rval






# test_input = torch.arange(4, dtype=torch.half)
buffer_size = 1
hidden_size = 128
max_sequence_length = 2048

test_input = torch.rand(1, buffer_size, 1, hidden_size, dtype=torch.half) * 0.25


torch_model = DynamicSizeArange(buffer_size, max_sequence_length, buffer_size, hidden_size)
inputs: List[ct.TensorType] = [
    ct.TensorType(shape=ct.Shape(shape=(1, buffer_size, 1, hidden_size)), name="input_array", dtype=np.half)
]

states: List[ct.StateType] = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(1024, max_sequence_length, buffer_size, 1, hidden_size), dtype=np.half),
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



print("Quantizing to int4")
op_config = ct.optimize.coreml.OpLinearQuantizerConfig(                                     
    mode="linear_symmetric",
    dtype="int4",                                                                           
    granularity="per_block",
    block_size=32,                                                                          
)
config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(mlmodel_fp16, config=config)
# mlmodel_int4._spec.description.metadata.userDefined.update({METADATA_TOKENIZER: MODEL_ID})
# mlmodel_int4.optimization_hints['reshapeFrequency'] = ct.ReshapeFrequency.Frequent
mlmodel_int4.save("realistic_int4.mlpackage")



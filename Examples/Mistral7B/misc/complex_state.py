import torch
import coremltools as ct
from typing import List
import numpy as np
from torch._C import dtype
        # self.register_buffer('counter', torch.zeros(1, dtype=torch.half))
        # make a cache shape 32, 32, 1, 128, 1, n
        # self.register_buffer('cache', torch.zeros(1, 128, 1, 32, dtype=torch.half))
        # self.positions = torch.arange(32, dtype=torch.half)

class DynamicSizeArange(torch.nn.Module):
    """
    A basic usage of cond based on dynamic shape predicate.
    """

    def __init__(self, size=8):
        super().__init__()
        self.register_buffer('counter', torch.tensor([0], dtype=torch.half))
        self.register_buffer('cache', torch.arange(32, dtype=torch.half))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.counter += x[0]
        self.cache += 0
        a = x.to(dtype=torch.float32)
        b = torch.mul(self.counter, self.cache)
        self.cache[:] = torch.concat((self.cache[1:], (a*a).to(torch.half))) # WORKS
        # self.cache = torch.concat((self.cache[1:], (a*a).to(torch.half))) # DOES NOT WORK
        return self.cache + 0

        # self.cache.add(1)
        # self.counter.add(1)
        # self.counter += 1
        # self.cache+=1
        # self.cache += 1
        # index = torch.tensor(x.shape[1])    # THIS WORKS
        # print(x.shape)
        # self.cache[:, :index] = x
        # index_copy()
        # self.cache.index_copy_(1, torch.arange(10), torch.arange(10))
        # self.cache[..., :x.shape[1]].index_copy_(3, torch.arange(x.shape[1]), x)

        # self.counter += 1
        # self.counter += 1
        # self.cache[:-1] = self.cache[1:]
        # self.cache[-1] = x
        # return x



        copied_cache = self.cache[1:].clone()
        self.cache[:] = torch.concat((copied_cache, x), dim=0)
        # WORKS

        # self.cache = torch.concat((self.cache[1:], x), dim=0)
        # WORKS
        return self.cache[-4:]

        self.cache


        # start = self.counter.to(torch.int)
        # start = self.counter.to(torch.int)
        # self.cache += 1
        # start = torch.round(self.counter)
        # print(start.shape)
        # end = torch.round(start + x.shape[-1])
        # return start
        # self.cache+=1 
        # index = torch.max(self.counter.to(torch.int), torch.tensor([1], dtype=torch.int)).to(torch.int)
        # return self.cache[..., torch.remainder(torch.arange(30) + 8, 32)] # SEEMS TO WORK
        # if self.counter == 3:
        #     return torch.arange(10)
        # return torch.arange(30)

        # if self.counter == 3:
        #     return torch.arange(30)
        # return torch.arange(10)


        return torch.cond(self.counter == 3, self.a, self.b, ())
        self.cache[..., [-2, -1, 0, 1]]  = x# SEEMS TO WORK
        return self.cache
        # start = torch.max(start, torch.tensor([1], dtype=torch.int))
       
        # return self.cache[..., :start.to(torch.int)]
        # self.cache[:, :, :, :start] = x # WORKS
        # self.counter += x.shape[-1]
        # self.cache[:, :, :, 0:x.shape[-1]].index_copy_(-1, torch.arange(x.shape[-1]), x) # DOES NOT WORK, NOT IMPLEMETNED
        # may just have to manually deal with assignment if basically writing over the end of 
        # self.cache[:, :, :, -2:2] = x # DOES NOT WORK
        return self.cache[:, :, :, :]
        # THIS WORKS W/ GPU ONLY AND ALL

        # self.cache[:, :, :, ]
        # self.cache[0, 0, 0, 0] = 1
        # self.cache.index_copy(1, torch.arange(index), x)
        # self.cache[:, :, :, torch.arange(index)] = x
        # self.cache.index_copy(-1, torch.arange(index), x)
        # self.cache[:, :, :, :index] =  1
        # self.cache[..., :index] = torch.concat((x, self.cache[:, :, :, index:]), dim=-1)
        return self.cache
        # self.cache[..., :index.to(torch.int)] += 1
        # return torch.tensor(x.shape[-1])    # THIS WORKS
        # return torch.remainder(self.counter, torch.tensor(x.shape[-1]))    # THIS WORKS
        # return torch.remainder(self.counter, torch.tensor([x.shape[-1]]))    # THIS DOES NOT
        return self.cache

# test_input = torch.arange(4, dtype=torch.half)
test_input = torch.tensor([0.001], dtype=torch.half)

buffer_size = 8
torch_model = DynamicSizeArange(buffer_size)
query_length = ct.RangeDim(lower_bound=1, upper_bound=32, default=1)
inputs: List[ct.TensorType] = [
    ct.TensorType(shape=ct.Shape(shape=(1,)), name="input_array", dtype=np.half)
]

states: List[ct.StateType] = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(32,), dtype=np.half),
        name="cache",
    ),
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(1, ), dtype=np.half),
        name="counter",
    ),
]
outputs: List[ct.TensorType] = [ct.TensorType(dtype=np.float32, name="logits")]
# torch_model.eval()
model = torch.jit.trace(torch_model, test_input)
mlmodel_fp16 = ct.convert(
        model,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        # skip_model_load=True,
        # compute_units=ct.ComputeUnit.ALL,     # WORKS
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        # compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
        debug=True,
        compute_precision=ct.transform.FP16ComputePrecision(),
    )
mlmodel_fp16.save("basicstate2.mlpackage")

state = mlmodel_fp16.make_state()

print(mlmodel_fp16.predict({"input_array": test_input.numpy()}, state))



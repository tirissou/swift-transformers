import torch

class DynamicSizeArange(torch.nn.Module):
    """
    A basic usage of cond based on dynamic shape predicate.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('counter', torch.zeros([1], dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        index = self.counter % x.shape
        rval = torch.concat((x[self.counter:], x[:self.counter]))
        self.counter += 1
        return rval

o = DynamicSizeArange()

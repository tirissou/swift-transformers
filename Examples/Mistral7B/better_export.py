import os
from itertools import cycle
import logging
from typing import List, Optional, Tuple
import coremltools as ct
from coremltools.models import MLModel
import numpy as np
import torch

from exfil import helpers
from exfil.mistral.model import StatefulMistralForCausalLM

# warnings.filterwarnings("ignore")
logging.getLogger("coremltools").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
MODEL_ID: str = "mistralai/Mistral-7B-Instruct-v0.3"
METADATA_TOKENIZER: str = "co.huggingface.exporters.name"


def export() -> None:
    # Construct model from transformers and trace to TorchScript
    max_context_size: int = 2048
    torch_model = StatefulMistralForCausalLM(MODEL_ID, max_context_size=max_context_size)
    torch_model.eval()

    # What inputs should the model have? 
    # Starting sequence, max additional tokens to generate
    input_ids: torch.Tensor = torch.tensor([[3429, 3458]], dtype=torch.long)
    max_additional_tokens = torch.tensor([128], dtype=torch.int32)
    helpers.replace_linear_with_conv2d(torch_model)
    
    import pdb
    pdb.set_trace()
    traced_model = torch.jit.trace(torch_model, [input_ids, max_additional_tokens])

    # Convert traced TorchScript to Core ML format
    print("Converting from TorchScript to Core ML format")
    query_length = ct.RangeDim(lower_bound=1, upper_bound=max_context_size, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=max_context_size, default=1)
    inputs: List[ct.TensorType] = [
        ct.TensorType(shape=(1, query_length), dtype=np.int32, name="inputIds"),
        ct.TensorType(
            shape=(1,),
            dtype=np.int32,
            name="maxNewTokens",
        )
    ]
    outputs: List[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]
    states: List[ct.StateType] = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=torch_model.kv_cache_shape, dtype=np.float16),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=torch_model.kv_cache_shape, dtype=np.float16),
            name="valueCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=(1,), dtype=np.int32),
            name="cacheSequenceLength",
        ),
        # ct.StateType(
        #     # TODO don't hardcode these shapes
        #     wrapped_type=ct.TensorType(shape=(32, 32, 1), dtype=np.long),
        #     name="cacheLength",
        # ),
        # ct.StateType(
        #     wrapped_type=ct.TensorType(shape=(32, 32, 1), dtype=np.long),
        #     name="cacheStartIndex",
        # ),
        # TODO add new states
    ]

    # Convert model with FP16 precision
    print("Converting with FP16 precision")
    mlmodel_fp16: MLModel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.transform.FP16ComputePrecision(),
    )


    # Block-wise quantize model weights to int4
    print("Quantizing to int4")
    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=32,
    )
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(mlmodel_fp16, config=config)
    mlmodel_int4._spec.description.metadata.userDefined.update({METADATA_TOKENIZER: MODEL_ID})
    mlmodel_int4.optimization_hints['reshapeFrequency'] = ct.ReshapeFrequency.Frequent
    mlmodel_int4.save("StatefulMistral7BInstructInt4.mlpackage")


if __name__ == "__main__":
    export()

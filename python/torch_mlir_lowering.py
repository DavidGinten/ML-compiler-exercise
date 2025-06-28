from typing import List

import torch
import torch.nn as nn
from torch.export import Dim
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import (
    make_boxed_compiler,
    get_aot_graph_name,
    set_model_name,
)

from torch_mlir import fx
from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
)

class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)

def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()

@run
def lower_pytorch_to_torch_fx():
    from torch.fx import symbolic_trace
    # Symbolic tracing frontend - captures the semantics of the module
    symbolic_traced: torch.fx.GraphModule = symbolic_trace(MyModule())

    # High-level intermediate representation (IR) - Graph representation
    print(symbolic_traced.graph)
"""
@run
def lower_pytorch_to_raw_output():
    # Export model to torch-mlir
    m = fx.export_and_import(MyModule(), torch.randn(3, 4), output_type=OutputType.LINALG_ON_TENSORS)

    # Model in torch dialect
    mlir_str = str(m)
    mlir_ir = mlir_str.split("{-#")[0].strip()
    print(mlir_ir)

    #with open("test_module_raw_mlir.txt", "w") as f:
    #    f.write(m)

@run
def lower_pytorch_to_torch_mlir():
    # Export model to torch-mlir
    m = fx.export_and_import(MyModule(), torch.randn(3, 4), output_type=OutputType.TORCH)

    # Model in torch dialect
    mlir_str = str(m)
    mlir_ir = mlir_str.split("{-#")[0].strip()
    print(mlir_ir)

    #with open("test_module_torch_mlir.txt", "w") as f:
    #    f.write(m)
"""

@run
def lower_pytorch_to_linalg_on_tensors():
    # Export model to torch-mlir
    m = fx.export_and_import(MyModule(), torch.randn(3, 4), output_type=OutputType.LINALG_ON_TENSORS)

    # Model in torch dialect
    mlir_str = str(m)
    mlir_ir = mlir_str.split("{-#")[0].strip()
    print(mlir_ir)

    with open("test_module_linalg_mlir.mlir", "w") as f:
        f.write(mlir_ir)
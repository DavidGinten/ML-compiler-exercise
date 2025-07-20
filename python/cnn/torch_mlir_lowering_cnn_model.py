from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
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

torch.manual_seed(41)

class ConvolutionalNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,6,3,1)
    self.conv2 = nn.Conv2d(6,16,3,1)
    # Fully Connected Layer
    self.fc1 = nn.Linear(5*5*16, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, X):
    X = F.relu(self.conv1(X))
    X = F.max_pool2d(X,2,2) # 2x2 kernal and stride 2
    # Second Pass
    X = F.relu(self.conv2(X))
    X = F.max_pool2d(X,2,2) # 2x2 kernal and stride 2

    # Re-View to flatten it out
    X = X.view(-1, 16*5*5) # negative one so that we can vary the batch size

    # Fully Connected Layers
    X = F.relu(self.fc1(X))
    X = F.relu(self.fc2(X))
    X = self.fc3(X)
    return F.log_softmax(X, dim=1)

def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()

"""
@run
def lower_pytorch_to_torch_fx():
    from torch.fx import symbolic_trace
    # Symbolic tracing frontend - captures the semantics of the module
    symbolic_traced: torch.fx.GraphModule = symbolic_trace(NeuralNetwork())

    # High-level intermediate representation (IR) - Graph representation
    print(symbolic_traced.graph)


@run
def lower_pytorch_to_raw_output():
    # Export model to torch-mlir
    m = fx.export_and_import(ConvolutionalNetwork(), torch.randn(1, 1, 28, 28), output_type=OutputType.RAW,
                             func_name = "cnn_model")

    # Model in torch dialect
    mlir_str = str(m)
    mlir_ir = mlir_str.split("{-#")[0].strip()
    print(mlir_ir)

    with open("cnn_model_raw.mlir", "w") as f:
        f.write(mlir_str)

@run
def lower_pytorch_to_torch_mlir():
    # Export model to torch-mlir
    m = fx.export_and_import(ConvolutionalNetwork(), torch.randn(1, 1, 28, 28), output_type=OutputType.TORCH,
                             func_name = "cnn_model")

    # Model in torch dialect
    mlir_str = str(m)
    mlir_ir = mlir_str.split("{-#")[0].strip()
    print(mlir_ir)

    with open("cnn_model_torch.mlir", "w") as f:
        f.write(mlir_str)
"""

@run
def lower_pytorch_to_linalg_on_tensors():
    # Export model to torch-mlir
    m = fx.export_and_import(ConvolutionalNetwork(), torch.randn(32, 1, 28, 28), output_type=OutputType.LINALG_ON_TENSORS, 
                             func_name = "cnn_model")
    
    # Model in torch dialect
    mlir_str = str(m)
    mlir_ir = mlir_str.split("{-#")[0].strip()
    #print(mlir_ir)

    with open("cnn_model_linalg.mlir", "w") as f:
        f.write(mlir_str)
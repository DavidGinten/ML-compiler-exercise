from torch_mlir.ir import Context, Location, Module, InsertionPoint
from torch_mlir.dialects import torch as torch_dialect

import torch
import torch.nn as nn
from torch.fx import symbolic_trace

class MyModel(nn.Module):
    def forward(self, x):
        return x * 2 + 1

model = MyModel()
traced = symbolic_trace(model)


# Helper: Convert FX op to MLIR op
def fx_node_to_torch_mlir_op(builder, node, value_map):
    if node.op == 'call_function':
        target = node.target

        if target == torch.mul:
            lhs = value_map[node.args[0]]
            rhs = value_map[node.args[1]]
            result = builder.aten_mul(lhs, rhs)
            value_map[node] = result
        elif target == torch.add:
            lhs = value_map[node.args[0]]
            rhs = value_map[node.args[1]]
            result = builder.aten_add(lhs, rhs)
            value_map[node] = result
        else:
            raise NotImplementedError(f"Unhandled op: {target}")
    elif node.op == 'placeholder':
        # Input tensor (function argument)
        # Simulate it as a `torch.tensor` type
        tensor_type = torch_dialect.TensorType.get(torch_dialect.UnknownType.get())
        value = builder.add_argument(tensor_type)
        value_map[node] = value
    elif node.op == 'output':
        result = value_map[node.args[0][0]]
        builder.return_op([result])
    else:
        raise NotImplementedError(f"Unhandled FX node type: {node.op}")

# Builder class to make MLIR construction easier
class TorchMlirBuilder:
    def __init__(self):
        self.context = Context()
        self.context.allow_unregistered_dialects = True
        self.module = Module.create()
        self.func = None
        self.ip = None

    def build(self, fx_graph):
        with self.context, self.module, InsertionPoint.at_block_begin(self.module.body):
            # Function type: (tensor) -> tensor
            input_type = torch_dialect.TensorType.get(torch_dialect.UnknownType.get())
            output_type = torch_dialect.TensorType.get(torch_dialect.UnknownType.get())
            func_type = torch_dialect.FunctionType.get([input_type], [output_type])
            self.func = torch_dialect.FuncOp("main", func_type)
            entry_block = self.func.add_entry_block()
            self.ip = InsertionPoint(entry_block)

            # Map FX nodes to MLIR values
            value_map = {}

            # Walk FX graph
            for node in fx_graph.graph.nodes:
                fx_node_to_torch_mlir_op(self, node, value_map)

        return self.module

    def add_argument(self, type_):
        return self.func.entry_block.arguments[0]

    def aten_mul(self, lhs, rhs):
        return torch_dialect.AtenMulTensorOp(lhs.type, lhs, rhs, None).result

    def aten_add(self, lhs, rhs):
        return torch_dialect.AtenAddTensorOp(lhs.type, lhs, rhs, None).result

    def return_op(self, values):
        return torch_dialect.ReturnOp(values)

# Use the builder
builder = TorchMlirBuilder()
mlir_module = builder.build(traced)
print(mlir_module)

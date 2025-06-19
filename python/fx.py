import torch
import torch.fx
import torch_mlir.ir
from torch_mlir.extras.fx_importer import FxImporter
from torch_mlir.extras.fx_decomp_util import get_decomposition_table
from torch_mlir.dialects import (
    torch as torch_dialect,
    func as func_dialect,
)

from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
    lower_mlir_module,
)

REQUIRED_DIALCTS = [
    "builtin",
    "func",
    "torch",
]

#####  Model definiton  ######

class MyModel(torch.nn.Module):
    #def __init__(self) -> None:
    #    super().__init__()
    #    self.param = torch.nn.Parameter(torch.rand(3, 4))
    #    self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return x * 2 + 1 #self.linear(x + self.param).clamp(min=0.0, max=1.0)

# (Assuming you have a PyTorch model and example_input from Phase 1)
f = MyModel()
example_input = (torch.randn(3),)
fx_graph_module = torch.fx.symbolic_trace(f)
fx_graph = fx_graph_module.graph

print(fx_graph)

class TorchMlirBuilder:
    __slots__ = [
        "_c",
        "_cc",
        "_m",
        "_m_ip",
        "_py_attr_tracker",
        "_hooks",
        "symbol_table",
    ]

    def __init__(
            self,
            *, # all following arguments must be explicitly named (keyword-only)
            context: Optional[Context] = None,
            config_check: bool = True,
    ):
        self._m = context if context else Context()
        self._m = Module.create(Location.unknown(self._c))
        if config_check:
            self._config_check()

    def _config_check(self):
        for dname in REQUIRED_DIALCTS:
            try:
                self._c.dialects[dname]
                logging.debug("Context has registered dialect '%s'", dname)
            except IndexError:
                raise RuntimeError(
                    f"The MLIR context {self._c} is missing required dialect '{dname}'"
                )
    
    @property
    def module(self) -> Module:
        return self._m

    def import_frozen_program(
            self,
            prog: torch.export.ExportedProgram,
            *,
            func_name: str = "main",
            func_visibility: Optional[str] = None,
            import_symbolic_shape_expressions: bool = False,
    ) -> Operation:
        """Imports a consolidated torch.export.ExportedProgram instance."""
        
        sig = prog.graph_signature
        state_dict = prog.state_dict
        arg_replacements: Dict[str, Any] = {}
        












def export_and_import():
    """ Here starts the fx.export_and_import function """
    context = torch_mlir.ir.Context()
    torch_dialect.register_dialect(context)

    builder = TorchMlirBuilder(context=context)

    # Export program
    prog = torch.export.export(
        f, example_input, {}, dynamic_shapes=None, strict=False
    )

    decomposition_table = get_decomposition_table()
    if decomposition_table:
        prog = prog.run_decompositions(decomposition_table)

    builder.import_frozen_program(
            prog,
            func_name="main",
            import_symbolic_shape_expressions=False,
    )
    return builder.module


if __name__ == "__main__":
    export_and_import()




"""
# Step 1: Create the MLIR Module and Function
with ctx, torch_mlir.ir.Location.unknown(): # Use an active context and a location
    mlir_module = torch_mlir.ir.Module.create()

    with torch_mlir.ir.InsertionPoint(mlir_module.body):
        # Determine the function signature from FX graph inputs/outputs
        # This will be more complex in a real implementation, inferring types
        # For simplicity, let's assume a single tensor input and output for now
        input_tensor_type = torch_mlir.ir.RankedTensorType.get(
            [1, 10], torch_mlir.ir.F32Type.get()
        )
        output_tensor_type = input_tensor_type # Assuming same for simple case

        func_type = torch_mlir.ir.FunctionType.get(
            inputs=[input_tensor_type],
            results=[output_tensor_type]
        )

        @func.func(name="main", input_mlir_value)
        def main_func(input_mlir_value: input_tensor_type) -> output_tensor_type:
        #def main_func(input_mlir_value): # This is our first MLIR SSA Value
            # Step 2: Initialize the mapping from FX Nodes to MLIR Values
            fx_node_to_mlir_value_map = {}

            # Map the placeholder node to the MLIR function argument
            # This is critical for connecting inputs
            for node in fx_graph.nodes:
                if node.op == 'placeholder':
                    # Assuming only one placeholder for simplicity
                    fx_node_to_mlir_value_map[node] =input_mlir_value
                    break

            # Step 3: Iterate through FX graph nodes in topological order
            # The .nodes property of an FX graph is already in topological order.
            for node in fx_graph.nodes:
                if node.op == 'placeholder' or node.op == 'output':
                    # Handled separately or at the end
                    continue

                # Get MLIR values for node inputs (operands)
                # This is where the fx_node_to_mlir_value_map is used
                mlir_operands = []
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        mlir_operands.append(fx_node_to_mlir_value_map[arg])
                    # Handle constant arguments if needed (e.g., node.args might have 1, 2.0 etc.)
                    elif isinstance(arg, (int, float)):
                        # For constants, you might emit a torch.prim.Constant op
                        constant_type = mlir.ir.F64Type.get() # or appropriate type
                        constant_val = mlir.ir.FloatAttr.get(constant_type, arg)
                        constant_op = torch_dialect.prim_ConstantOp(constant_val) # Assuming this op exists
                        mlir_operands.append(constant_op.result)
                    else:
                        raise NotImplementedError(f"Unhandled arg type: {type(arg)} for node {node}")

                # Handle keyword arguments if needed (node.kwargs)
                mlir_attributes = {}
                for kwarg_name, kwarg_value in node.kwargs.items():
                    # Map FX kwargs to MLIR attributes if necessary
                    # This depends on the specific MLIR op.
                    pass # Simplified for this example

                # Step 4: Emit the corresponding MLIR operation
                # This is the core mapping logic!
                emitted_mlir_result = None
                if node.target == torch.ops.aten.add.Tensor:
                    # torch_dialect.aten_add.Tensor is a placeholder for the actual op
                    # The name might be different, you'd check MLIR source or docs
                    # mlir_operands should be [lhs, rhs]
                    emitted_mlir_result = torch_dialect.aten_add(
                        mlir_operands[0], mlir_operands[1]
                    ).result
                elif node.target == torch.ops.aten.relu.default:
                    emitted_mlir_result = torch_dialect.aten_relu(
                        mlir_operands[0]
                    ).result
                # ... add more mappings for other torch.ops.aten.* functions
                else:
                    raise NotImplementedError(f"Unhandled FX node target: {node.target}")

                # Step 5: Store the result in the map for future nodes
                fx_node_to_mlir_value_map[node] = emitted_mlir_result

            # Step 6: Handle the 'output' node
            # The output node's args are the values to return.
            output_node = next(n for n in fx_graph.nodes if n.op == 'output')
            return_mlir_values = []
            for arg in output_node.args[0]: # Output node usually has a tuple of results in args[0]
                return_mlir_values.append(fx_node_to_mlir_value_map[arg])
            func.Return(return_mlir_values)

    # Step 7: Print the generated MLIR Module
    print(mlir_module)
    """

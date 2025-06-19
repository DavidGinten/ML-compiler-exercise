// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.mm$basic(
// CHECK-SAME:                        %[[LHS_VTENSOR:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                        %[[RHS_VTENSOR:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,2],f32> {
// CHECK-DAG:       %[[LHS:.*]] = torch_c.to_builtin_tensor %[[LHS_VTENSOR]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:       %[[RHS:.*]] = torch_c.to_builtin_tensor %[[RHS_VTENSOR]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[LHS_DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[RHS_DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[LHS_DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[RHS_DIM_0:.*]] = tensor.dim %[[RHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[EQ:.*]] = arith.cmpi eq, %[[LHS_DIM_1]], %[[RHS_DIM_0]] : index
// CHECK:           assert %[[EQ]], "mismatching contracting dimension for torch.aten.mm"
// CHECK:           %[[INIT_TENSOR:.*]] = tensor.empty(%[[LHS_DIM_0]], %[[RHS_DIM_1]]) : tensor<?x?xf32>
// CHECK:           %[[CF0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[ZEROFILL:.*]] = linalg.fill ins(%[[CF0]] : f32) outs(%[[INIT_TENSOR]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[MATMUL:.*]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ZEROFILL]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[CASTED:.*]] = tensor.cast %[[MATMUL]] : tensor<?x?xf32> to tensor<?x2xf32>
// CHECK:           %[[RESULT_VTENSOR:.*]] = torch_c.from_builtin_tensor %[[CASTED]] : tensor<?x2xf32> -> !torch.vtensor<[?,2],f32>
// CHECK:           return %[[RESULT_VTENSOR]] : !torch.vtensor<[?,2],f32>
func.func @torch.aten.mm$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,2],f32> {
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,2],f32>
  return %0 : !torch.vtensor<[?,2],f32>
}

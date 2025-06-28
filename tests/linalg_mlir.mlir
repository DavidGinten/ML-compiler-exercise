#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<3x4xf32>) -> tensor<3x5xf32> {
    %cst = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant dense_resource<torch_tensor_5_4_torch.float32> : tensor<5x4xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_5_torch.float32> : tensor<5xf32>
    %0 = tensor.empty() : tensor<3x4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst : tensor<3x4xf32>, tensor<3x4xf32>) outs(%0 : tensor<3x4xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %8 = arith.addf %in, %in_4 : f32
      linalg.yield %8 : f32
    } -> tensor<3x4xf32>
    %2 = tensor.empty() : tensor<4x5xf32>
    %transposed = linalg.transpose ins(%cst_2 : tensor<5x4xf32>) outs(%2 : tensor<4x5xf32>) permutation = [1, 0] 
    %3 = tensor.empty() : tensor<3x5xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<3x5xf32>) -> tensor<3x5xf32>
    %5 = linalg.matmul ins(%1, %transposed : tensor<3x4xf32>, tensor<4x5xf32>) outs(%4 : tensor<3x5xf32>) -> tensor<3x5xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %cst_3 : tensor<3x5xf32>, tensor<5xf32>) outs(%3 : tensor<3x5xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %8 = arith.addf %in, %in_4 : f32
      linalg.yield %8 : f32
    } -> tensor<3x5xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<3x5xf32>) outs(%3 : tensor<3x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.cmpf ult, %in, %cst_0 : f32
      %9 = arith.select %8, %cst_0, %in : f32
      %10 = arith.cmpf ugt, %9, %cst_1 : f32
      %11 = arith.select %10, %cst_1, %9 : f32
      linalg.yield %11 : f32
    } -> tensor<3x5xf32>
    return %7 : tensor<3x5xf32>
  }
}
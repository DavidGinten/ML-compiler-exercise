module {
  func.func @test_poly_fn(%arg0: i32) -> i32 {
    %c11 = arith.constant 11 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0> : tensor<10xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<[2, 3, 4]> : tensor<3xi32>
    %splat = tensor.splat %arg0 : tensor<10xi32>
    %padded = tensor.pad %cst_0 low[0] high[7] {
    ^bb0(%arg1: index):
      tensor.yield %c0_i32 : i32
    } : tensor<3xi32> to tensor<10xi32>
    %0 = arith.addi %padded, %splat : tensor<10xi32>
    %1 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %cst) -> (tensor<10xi32>) {
      %4 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %arg2) -> (tensor<10xi32>) {
        %5 = arith.addi %arg1, %arg3 : index
        %6 = arith.remui %5, %c10 : index
        %extracted = tensor.extract %0[%arg3] : tensor<10xi32>
        %extracted_1 = tensor.extract %0[%arg1] : tensor<10xi32>
        %7 = arith.muli %extracted_1, %extracted : i32
        %extracted_2 = tensor.extract %arg4[%6] : tensor<10xi32>
        %8 = arith.addi %7, %extracted_2 : i32
        %inserted = tensor.insert %8 into %arg4[%6] : tensor<10xi32>
        scf.yield %inserted : tensor<10xi32>
      }
      scf.yield %4 : tensor<10xi32>
    }
    %2 = arith.subi %1, %splat : tensor<10xi32>
    %3 = scf.for %arg1 = %c1 to %c11 step %c1 iter_args(%arg2 = %c0_i32) -> (i32) {
      %4 = arith.subi %c11, %arg1 : index
      %5 = arith.muli %arg0, %arg2 : i32
      %extracted = tensor.extract %2[%4] : tensor<10xi32>
      %6 = arith.addi %5, %extracted : i32
      scf.yield %6 : i32
    }
    return %3 : i32
  }
}
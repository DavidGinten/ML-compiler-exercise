// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(canonicalize{test-convergence}))' -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_subi_zero
func.func @test_subi_zero(%arg0: i32) -> i32 {
  // CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
  // CHECK-NEXT: return %c0
  %y = arith.subi %arg0, %arg0 : i32
  return %y: i32
}

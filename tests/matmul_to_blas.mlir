// RUN: tutorial-opt <%s -linalg-to-bufferization -convert-matmul-to-blas -split-input-file -verify-diagnostics | FileCheck %s

func.func @foo(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul
    ins(%lhs, %rhs: tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%acc: tensor<?x?xf32>)
  -> tensor<?x?xf32>
  return %result: tensor<?x?xf32>
}
#ifndef LIB_CONVERTMATMULTOBLAS_H
#define LIB_CONVERTMATMULTOBLAS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

// Custom pass to replace linalg.matmul with OpenBLAS calls
struct ConvertMatmulToBlasLibraryCallPass
    : public PassWrapper<ConvertMatmulToBlasLibraryCallPass,
                         OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConvertMatmulToBlasLibraryCallPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, func::FuncDialect, memref::MemRefDialect,
                    arith::ArithDialect>();
  }

private:
  void runOnOperation() override;

  StringRef getArgument() const final { return "convert-matmul-to-blas"; }

  StringRef getDescription() const final {
    return "Convert linalg.matmul operations to CBLAS function calls";
  }
};

} // namespace tutorial
} // namespace mlir

#endif // LIB_CONVERTMATMULTOBLAS_H
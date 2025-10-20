#include "lib/ConvertMatMulToBlas.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

// Create the pass
std::unique_ptr<Pass> createConvertMatmulToBlasLibraryCallPass() {
  return std::make_unique<mlir::tutorial::ConvertMatmulToBlasLibraryCallPass>();
}

void linalgToBufferizationPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // Linalg optimizations (but keep matmuls for BLAS replacement)
  //manager.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
  //manager.addPass(mlir::createLinalgElementwiseOpFusionPass());

  // One-shot bufferize
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager, deallocationOptions);
}

void BufferizationToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // CRITICAL: Replace matmuls with BLAS calls AFTER bufferization but BEFORE other LLVM conversions
  manager.addPass(createConvertMatmulToBlasLibraryCallPass());

  // Convert remaining linalg ops to loops
  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  // Standard LLVM lowering pipeline
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());
  manager.addPass(mlir::createLowerAffinePass());
  manager.addPass(mlir::affine::createLoopFusionPass());
  manager.addPass(mlir::affine::createAffineVectorize());
  manager.addPass(mlir::createSCFToControlFlowPass());
  
  // Convert to LLVM - order matters here
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertMathToLLVMPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Cleanup
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::PassRegistration<mlir::tutorial::ConvertMatmulToBlasLibraryCallPass>();

  mlir::PassPipelineRegistration<>("linalg-to-bufferization",
                             "Run passes to lower the linalg dialect to bufferization",
                             linalgToBufferizationPipelineBuilder);
                      
  mlir::PassPipelineRegistration<>("bufferization-to-llvm",
                             "Run passes to lower bufferized code to LLVM",
                             BufferizationToLLVMPipelineBuilder);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
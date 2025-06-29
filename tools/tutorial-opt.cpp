#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

void linalgToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::createCanonicalizerPass());
  //mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
  //bufferizationOptions.bufferizeFunctionBoundaries = true;
  
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass());
  //mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  //mlir::bufferization::buildBufferDeallocationPipeline(manager, deallocationOptions);

  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  manager.addPass(mlir::createCanonicalizerPass());

  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  //registry.insert<mlir::tutorial::poly::PolyDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  // Dialect conversion passes
  //mlir::tutorial::poly::registerPolyToStandardPasses();

  mlir::PassPipelineRegistration<>("linalg-to-llvm",
                             "Run passes to lower the linalg dialect to LLVM",
                             linalgToLLVMPipelineBuilder);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}

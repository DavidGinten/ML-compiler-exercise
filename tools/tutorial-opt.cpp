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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

// Custom conversion pattern for linalg.matmul to BLAS calls
struct MatmulToBlasPattern : public ConversionPattern {
  explicit MatmulToBlasPattern(MLIRContext *context)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto matmulOp = cast<linalg::MatmulOp>(op);
    Location loc = matmulOp.getLoc();
    
    // Check if this is a 2D matmul on f32 memrefs
    auto lhsType = cast<MemRefType>(matmulOp.getInputs()[0].getType()); // matmulOp.getInputs()[0].getType().dyn_cast<MemRefType>();
    auto rhsType = cast<MemRefType>(matmulOp.getInputs()[1].getType()); // matmulOp.getInputs()[1].getType().dyn_cast<MemRefType>();
    auto outputType = cast<MemRefType>(matmulOp.getInputs()[2].getType()); // matmulOp.getOutputs()[0].getType().dyn_cast<MemRefType>();
    
    if (!lhsType || !rhsType || !outputType ||
        lhsType.getRank() != 2 || rhsType.getRank() != 2 || outputType.getRank() != 2 ||
        !lhsType.getElementType().isF32()) {
      return failure();
    }
    
    // Get the operands after bufferization
    Value lhs = operands[0];
    Value rhs = operands[1]; 
    Value output = operands[2];
    
    // Extract matrix dimensions
    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();
    auto outputShape = outputType.getShape();
    
    // Create LLVM types
    auto i32Type = IntegerType::get(rewriter.getContext(), 32);
    auto f32Type = Float32Type::get(rewriter.getContext());
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    // Get or create the cblas_sgemm function declaration
    ModuleOp module = matmulOp->getParentOfType<ModuleOp>();
    LLVM::LLVMFuncOp sgemmFunc = getOrCreateSgemmFunc(module, rewriter);
    
    // Create constants for cblas_sgemm parameters
    Value order = rewriter.create<LLVM::ConstantOp>(loc, i32Type, 
                                                    rewriter.getI32IntegerAttr(101)); // CblasRowMajor
    Value transA = rewriter.create<LLVM::ConstantOp>(loc, i32Type, 
                                                     rewriter.getI32IntegerAttr(111)); // CblasNoTrans
    Value transB = rewriter.create<LLVM::ConstantOp>(loc, i32Type, 
                                                     rewriter.getI32IntegerAttr(111)); // CblasNoTrans
    
    // Matrix dimensions - handle both static and dynamic shapes
    Value M, N, K, ldA, ldB, ldC;
    
    if (lhsShape[0] != ShapedType::kDynamic) {
      M = rewriter.create<LLVM::ConstantOp>(loc, i32Type, 
                                            rewriter.getI32IntegerAttr(lhsShape[0]));
    } else {
      Value dimM = rewriter.create<memref::DimOp>(loc, lhs, 0);
      M = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimM);
    }
    
    if (rhsShape[1] != ShapedType::kDynamic) {
      N = rewriter.create<LLVM::ConstantOp>(loc, i32Type, 
                                            rewriter.getI32IntegerAttr(rhsShape[1]));
    } else {
      Value dimN = rewriter.create<memref::DimOp>(loc, rhs, 1);
      N = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimN);
    }
    
    if (lhsShape[1] != ShapedType::kDynamic) {
      K = rewriter.create<LLVM::ConstantOp>(loc, i32Type, 
                                            rewriter.getI32IntegerAttr(lhsShape[1]));
      ldA = rewriter.create<LLVM::ConstantOp>(loc, i32Type, 
                                              rewriter.getI32IntegerAttr(lhsShape[1]));
    } else {
      Value dimK = rewriter.create<memref::DimOp>(loc, lhs, 1);
      K = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimK);
      ldA = K;
    }
    
    if (rhsShape[1] != ShapedType::kDynamic) {
      ldB = rewriter.create<LLVM::ConstantOp>(loc, i32Type, 
                                              rewriter.getI32IntegerAttr(rhsShape[1]));
    } else {
      Value dimLdB = rewriter.create<memref::DimOp>(loc, rhs, 1);
      ldB = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimLdB);
    }
    
    if (outputShape[1] != ShapedType::kDynamic) {
      ldC = rewriter.create<LLVM::ConstantOp>(loc, i32Type, 
                                              rewriter.getI32IntegerAttr(outputShape[1]));
    } else {
      Value dimLdC = rewriter.create<memref::DimOp>(loc, output, 1);
      ldC = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimLdC);
    }
    
    // Alpha and Beta scalars
    Value alpha = rewriter.create<LLVM::ConstantOp>(loc, f32Type, 
                                                    rewriter.getF32FloatAttr(1.0));
    Value beta = rewriter.create<LLVM::ConstantOp>(loc, f32Type, 
                                                   rewriter.getF32FloatAttr(0.0));
    
    // Extract pointers from memrefs
    Value lhsPtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, lhs);
    Value rhsPtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, rhs);
    Value outputPtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, output);
    
    // Convert to LLVM pointers
    lhsPtr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, lhsPtr);
    rhsPtr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, rhsPtr);
    outputPtr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, outputPtr);
    
    // Create the function call
    SmallVector<Value> args = {order, transA, transB, M, N, K, alpha, 
                               lhsPtr, ldA, rhsPtr, ldB, beta, outputPtr, ldC};
    
    rewriter.create<LLVM::CallOp>(loc, sgemmFunc, args);
    
    // Erase the original matmul operation
    rewriter.eraseOp(matmulOp);
    
    return success();
  }

private:
  LLVM::LLVMFuncOp getOrCreateSgemmFunc(ModuleOp module, PatternRewriter &rewriter) const {
    const StringRef funcName = "cblas_sgemm";
    
    // Check if function already exists
    if (auto existingFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
      return existingFunc;
    }
    
    // Create function type for cblas_sgemm
    auto i32Type = IntegerType::get(rewriter.getContext(), 32);
    auto f32Type = Float32Type::get(rewriter.getContext());
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    SmallVector<Type> argTypes = {
      i32Type,  // Order
      i32Type,  // TransA
      i32Type,  // TransB
      i32Type,  // M
      i32Type,  // N
      i32Type,  // K
      f32Type,  // alpha
      ptrType,  // A
      i32Type,  // lda
      ptrType,  // B
      i32Type,  // ldb
      f32Type,  // beta
      ptrType,  // C
      i32Type   // ldc
    };
    
    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), argTypes);
    
    // Create function declaration
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    
    auto sgemmFunc = rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), funcName, funcType);
    sgemmFunc.setPrivate();
    
    return sgemmFunc;
  }
};

// Custom pass to replace linalg.matmul with OpenBLAS calls
struct LinalgToBlasPass : public PassWrapper<LinalgToBlasPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToBlasPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, memref::MemRefDialect, 
                   LLVM::LLVMDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    
    // Set up type converter for partial LLVM conversion
    LLVMTypeConverter typeConverter(context);
    
    RewritePatternSet patterns(context);
    patterns.add<MatmulToBlasPattern>(context);
    
    ConversionTarget target(*context);
    target.addLegalDialect<func::FuncDialect, memref::MemRefDialect, 
                          arith::ArithDialect, LLVM::LLVMDialect>();
    target.addIllegalOp<linalg::MatmulOp>();
    
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

// Create the pass
std::unique_ptr<Pass> createLinalgToBlasPass() {
  return std::make_unique<LinalgToBlasPass>();
}

void linalgToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  /* For BLAS
  // Linalg optimizations (but keep matmuls for BLAS replacement)
  manager.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
  manager.addPass(mlir::createLinalgElementwiseOpFusionPass());

  manager.addPass(mlir::createCanonicalizerPass());
  //manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  //manager.addPass(mlir::createConvertTensorToLinalgPass());
  manager.addPass(createLinalgToBlasPass());

  */

  // One-shot bufferize
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager, deallocationOptions);
  
  /* For BLAS*/
  // CRITICAL: Replace matmuls with BLAS calls AFTER bufferization
  //manager.addPass(createLinalgToBlasPass());

  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  // Needed to lower memref.subview
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());
  
  manager.addPass(mlir::createLowerAffinePass());
  manager.addPass(mlir::affine::createLoopFusionPass());
  manager.addPass(mlir::affine::createAffineVectorize());
  manager.addPass(mlir::createSCFToControlFlowPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertMathToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());

  // Cleanup
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
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
/*
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
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

void optimizedLinalgToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // Initial canonicalization
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // CRITICAL: Linalg-level optimizations BEFORE bufferization
  manager.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
  
  // Tiling for better cache locality and vectorization
  // Adjust tile sizes based on your target architecture
  manager.addPass(mlir::createForallToParallelLoopPass());
  
  // Fusion at Linalg level - crucial for CNN performance
  manager.addPass(mlir::createLinalgFoldReshapeOpsByLinearizationPass());
  manager.addPass(mlir::createLinalgElementwiseOpFusionPass());
  
  // More canonicalization after transformations
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createCSEPass());

  // Vectorization at Linalg level (much more effective than later)
  manager.addPass(mlir::createLinalgVectorizePass());
  
  // Vector-level optimizations
  manager.addPass(mlir::vector::createVectorTransferFullPartialRewritePass());
  manager.addPass(mlir::vector::createVectorTransferDropUnitDimsPass());
  manager.addPass(mlir::vector::createVectorTransferFlattenPass());
  
  // Bufferization
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  bufferizationOptions.allowReturnAllocsFromLoops = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager, deallocationOptions);

  // Post-bufferization vector optimizations
  manager.addPass(mlir::vector::createVectorBufferizationPass());
  manager.addPass(mlir::vector::createVectorTransferOptimizationPass());
  manager.addPass(mlir::vector::createVectorContractLoweringPass());
  manager.addPass(mlir::vector::createVectorMultiReductionLoweringPass());
  manager.addPass(mlir::vector::createVectorTransferLoweringPass());
  manager.addPass(mlir::vector::createVectorShapeCastLoweringPass());
  
  // Convert remaining Linalg ops to loops (only what's left after vectorization)
  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  // Memref optimizations
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());
  manager.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  
  // Affine optimizations
  manager.addPass(mlir::createLowerAffinePass());
  manager.addPass(mlir::affine::createLoopFusionPass());
  manager.addPass(mlir::affine::createAffineLoopInvariantCodeMotionPass());
  manager.addPass(mlir::affine::createAffineScalarReplacementPass());
  
  // Final cleanup before LLVM lowering
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createCSEPass());
  
  // Lower to LLVM
  manager.addPass(mlir::createConvertVectorToLLVMPass());
  manager.addPass(mlir::createSCFToControlFlowPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertMathToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());

  // Final cleanup
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}

// Alternative pipeline with explicit tiling configuration
void tiledLinalgToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // Explicit tiling with sizes optimized for CNN workloads
  // You may need to adjust these based on your specific architecture
  auto tilingOptions = mlir::linalg::LinalgTilingOptions();
  tilingOptions.setTileSizes({32, 32, 8}); // Adjust for your conv sizes
  manager.addPass(mlir::createLinalgTilePass(tilingOptions));
  
  // Continue with the rest of the optimized pipeline...
  // (same as above from Linalg optimizations onward)
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::PassPipelineRegistration<>("optimized-linalg-to-llvm",
                             "Optimized pipeline for CNN compilation",
                             optimizedLinalgToLLVMPipelineBuilder);

  mlir::PassPipelineRegistration<>("tiled-linalg-to-llvm",
                             "Tiled pipeline for CNN compilation",
                             tiledLinalgToLLVMPipelineBuilder);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Optimized CNN Pass Driver", registry));
}*/
#include "lib/ConvertMatMulToBlas.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/Passes.h"
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

namespace mlir {
namespace tutorial {

// Pattern to convert Linalg Matmul Op to cBlas function call.
struct MatmulOpToBlasLibraryCall : public ConversionPattern {
  explicit MatmulOpToBlasLibraryCall(MLIRContext *context,
                                     TypeConverter &converter)
      : ConversionPattern(converter, linalg::MatmulOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto matmulOp = cast<linalg::MatmulOp>(op);
    Location loc = matmulOp.getLoc();

    // Debug: Print information about the operation
    llvm::errs() << "Processing matmul with " << operands.size()
                 << " operands\n";
    llvm::errs() << "Number of inputs: " << matmulOp.getNumDpsInputs() << "\n";
    llvm::errs() << "Number of outputs: " << matmulOp.getNumDpsInits() << "\n";

    // Get inputs and outputs using the proper accessors
    auto inputs = matmulOp.getDpsInputs();
    auto outputs = matmulOp.getDpsInits();

    if (inputs.size() != 2 || outputs.size() != 1) {
      llvm::errs() << "Unexpected number of inputs/outputs\n";
      return failure();
    }

    // Get ORIGINAL operands (before conversion) to check types
    Value originalLhs = inputs[0];
    Value originalRhs = inputs[1];
    Value originalOutput = outputs[0];

    // Check if this is a 2D matmul on f32 memrefs using ORIGINAL types
    auto lhsType = dyn_cast<MemRefType>(originalLhs.getType());
    auto rhsType = dyn_cast<MemRefType>(originalRhs.getType());
    auto outputType = dyn_cast<MemRefType>(originalOutput.getType());

    llvm::errs() << "Original LHS type: " << lhsType << "\n";
    llvm::errs() << "Original RHS type: " << rhsType << "\n";
    llvm::errs() << "Original Output type: " << outputType << "\n";

    if (!lhsType || !rhsType || !outputType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2 || outputType.getRank() != 2 ||
        !lhsType.getElementType().isF32()) {
      llvm::errs() << "Type check failed\n";
      return failure();
    }

    llvm::errs() << "Type check passed\n";

    // Now get the converted operands (these should be LLVM struct types)
    Value lhs = operands[0];    // First input (converted)
    Value rhs = operands[1];    // Second input (converted)
    Value output = operands[2]; // Output (converted)

    llvm::errs() << "Converted LHS type: " << lhs.getType() << "\n";
    llvm::errs() << "Converted RHS type: " << rhs.getType() << "\n";
    llvm::errs() << "Converted Output type: " << output.getType() << "\n";

    // Extract matrix dimensions from ORIGINAL types
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
    Value order = rewriter.create<LLVM::ConstantOp>(
        loc, i32Type,
        rewriter.getI32IntegerAttr(101)); // CblasRowMajor
    Value transA = rewriter.create<LLVM::ConstantOp>(
        loc, i32Type,
        rewriter.getI32IntegerAttr(111)); // CblasNoTrans
    Value transB = rewriter.create<LLVM::ConstantOp>(
        loc, i32Type,
        rewriter.getI32IntegerAttr(111)); // CblasNoTrans

    // Matrix dimensions - handle both static and dynamic shapes
    Value M, N, K, ldA, ldB, ldC;

    if (lhsShape[0] != ShapedType::kDynamic) {
      M = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(lhsShape[0]));
    } else {
      // Use original operand for dimension extraction, then convert to i32
      Value dimM = rewriter.create<memref::DimOp>(loc, originalLhs, 0);
      M = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimM);
    }

    if (rhsShape[1] != ShapedType::kDynamic) {
      N = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(rhsShape[1]));
    } else {
      Value dimN = rewriter.create<memref::DimOp>(loc, originalRhs, 1);
      N = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimN);
    }

    if (lhsShape[1] != ShapedType::kDynamic) {
      K = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(lhsShape[1]));
      ldA = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(lhsShape[1]));
    } else {
      Value dimK = rewriter.create<memref::DimOp>(loc, originalLhs, 1);
      K = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimK);
      ldA = K;
    }

    if (rhsShape[1] != ShapedType::kDynamic) {
      ldB = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(rhsShape[1]));
    } else {
      Value dimLdB = rewriter.create<memref::DimOp>(loc, originalRhs, 1);
      ldB = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimLdB);
    }

    if (outputShape[1] != ShapedType::kDynamic) {
      ldC = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(outputShape[1]));
    } else {
      Value dimLdC = rewriter.create<memref::DimOp>(loc, originalOutput, 1);
      ldC = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimLdC);
    }

    // Alpha and Beta scalars
    Value alpha = rewriter.create<LLVM::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(1.0));
    Value beta = rewriter.create<LLVM::ConstantOp>(
        loc, f32Type, rewriter.getF32FloatAttr(0.0));

    // Extract pointers from memrefs using LLVM operations
    Value lhsPtr =
        rewriter.create<LLVM::ExtractValueOp>(loc, lhs, ArrayRef<int64_t>{1});
    Value rhsPtr =
        rewriter.create<LLVM::ExtractValueOp>(loc, rhs, ArrayRef<int64_t>{1});
    Value outputPtr = rewriter.create<LLVM::ExtractValueOp>(
        loc, output, ArrayRef<int64_t>{1});

    // Create the function call
    SmallVector<Value> args = {order, transA, transB,    M,   N,
                               K,     alpha,  lhsPtr,    ldA, rhsPtr,
                               ldB,   beta,   outputPtr, ldC};

    rewriter.create<LLVM::CallOp>(loc, sgemmFunc, args);

    // Erase the original matmul operation
    rewriter.eraseOp(matmulOp);

    return success();
  }

private:
  LLVM::LLVMFuncOp getOrCreateSgemmFunc(ModuleOp module,
                                        PatternRewriter &rewriter) const {
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
        i32Type, // Order
        i32Type, // TransA
        i32Type, // TransB
        i32Type, // M
        i32Type, // N
        i32Type, // K
        f32Type, // alpha
        ptrType, // A
        i32Type, // lda
        ptrType, // B
        i32Type, // ldb
        f32Type, // beta
        ptrType, // C
        i32Type  // ldc
    };

    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), argTypes);

    // Create function declaration
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());

    auto sgemmFunc = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                       funcName, funcType);
    sgemmFunc.setPrivate();

    return sgemmFunc;
  }
};

// Custom pass to replace linalg.matmul with OpenBLAS calls
void ConvertMatmulToBlasLibraryCallPass::runOnOperation() {
  auto &context = getContext();
  ConversionTarget target(context);

  Operation *op = getOperation();

  llvm::errs() << "Running ConvertMatmulToBlasLibraryCallPass\n";

  // Mark legal dialects
  target.addLegalDialect<func::FuncDialect, LLVM::LLVMDialect,
                         memref::MemRefDialect, arith::ArithDialect>();

  // Mark linalg.matmul as illegal - this forces the conversion
  target.addIllegalOp<linalg::MatmulOp>();

  RewritePatternSet patterns(&context);
  LLVMTypeConverter typeConverter(&context);
  patterns.add<MatmulOpToBlasLibraryCall>(patterns.getContext(), typeConverter);

  llvm::errs() << "About to apply conversion\n";
  if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
    llvm::errs() << "Conversion failed\n";
    signalPassFailure();
  } else {
    llvm::errs() << "Conversion succeeded\n";
  }
}

} // namespace tutorial
} // namespace mlir
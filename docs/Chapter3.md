# Lowering models to machine code

As in the tutorials, we will use existing passes to lower our models to llvm exit dialect. We create out own pipeline with all the passes, so at the end we only have to run pass that runs them all. This pipeline is defined in tools/tutorials-opt.cpp. There might be different combinations of passes you can run. We add the following passes:

``` C++
void linalgToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // One-shot bufferize
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager, deallocationOptions);

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
```
In an upcoming chapter, we will also use –llvm-request-c-wrappers (you can also add the attribute as in the tutorials) at the beginning that emits a C-friendly callable function that is used to call the model from C later on. Wirth that, we can pass a pointer to the memref struct instead of passing all 5 entries individual. But here we don't use it to illustrate both varients and their difference. 

We can register the pipeline in main() with
``` C++
  mlir::PassPipelineRegistration<>("linalg-to-llvm",
                             "Run passes to lower the linalg dialect to LLVM",
                             linalgToLLVMPipelineBuilder);
```

After this pipeline of passes we have our model in MLIR llvm dialect available. We can then translate the code to llvm ir, leaving the MLIR space:

`mlir-translate -mlir-to-llvmir $PWD/sample_model_llvm.mlir > $PWD/sample_model_llvm_ir.ll`

Then create the object file:
`llc --filetype=obj $PWD/sample_model_llvm_ir.ll`


In our C++ program (sample_model_main.cpp), we initalize the input tensor, set offset, size, and stride. We can then call the sample_model. A memref is returend where we access the data. With the correct memory accessing (consider size and stride) we can print the output.    

For that, we define a memref struct 
``` C++
    typedef struct {
        void* allocated;
        void* aligned;
        int64_t offset;
        int64_t sizes[2];
        int64_t strides[2];
    } MemRef2D;
```

The function signature of our model looks as follows:
``` C++
    MemRef2D sample_model(
        void* allocated,
        void* aligned,
        int64_t offset,
        int64_t size0,
        int64_t size1,
        int64_t stride0,
        int64_t stride1
    );
```

Compile and link the code:
`gcc -c sample_model_main.cpp -o sample_model_main.o && gcc sample_model_main.o sample_model_llvm_ir.o -o a.out`

Run: `./a.out`

The same procedure is also applied for the other dummy models: mnist and cnn
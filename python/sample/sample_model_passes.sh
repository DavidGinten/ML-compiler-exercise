###  Pipeline to get from linalg to llvm dialect  ###
python torch_mlir_lowering_sample_model.py

#mlir-opt --canonicalize --convert-elementwise-to-linalg --convert-tensor-to-linalg --one-shot-bufferize=bufferize-function-boundaries --buffer-deallocation-pipeline --convert-linalg-to-loops --expand-strided-metadata --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts --convert-func-to-llvm --canonicalize --sccp --cse --symbol-dce $PWD/python/sample_model_linalg.mlir

# Legacy
../../build-ninja/tools/tutorial-opt --linalg-to-llvm $PWD/sample_model_linalg.mlir > $PWD/sample_model_llvm.mlir

# With Blas integration (Todo: merge)
../../build-ninja/tools/tutorial-opt --linalg-to-bufferization $PWD/sample_model_linalg.mlir > $PWD/sample_model_buf_linalg.mlir
../../build-ninja/tools/tutorial-opt --bufferization-to-llvm $PWD/sample_model_buf_linalg.mlir > $PWD/sample_model_llvm.mlir

###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir $PWD/sample_model_llvm.mlir > $PWD/sample_model_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj $PWD/sample_model_llvm_ir.ll

###  Compile  ###
#gcc -c sample_model_main.cpp -o sample_model_main.o && gcc sample_model_main.o sample_model_llvm_ir.o -o a.out -lm

gcc -c sample_model_main.cpp -o sample_model_main.o && gcc sample_model_main.o sample_model_llvm_ir.o -o a.out
gcc -c sample_model_main.cpp -o sample_model_main.o && gcc sample_model_main.o sample_model_llvm_ir.o -o a.out -L../../lib -lopenblas -lm

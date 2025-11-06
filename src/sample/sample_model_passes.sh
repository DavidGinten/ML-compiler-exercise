###  Pipeline to get from linalg to llvm dialect  ###
python torch_mlir_lowering_sample_model.py

# With BLAS integration (Todo: merge)
../../build-ninja/tools/tutorial-opt --linalg-to-bufferization $PWD/sample_model_linalg.mlir > $PWD/sample_model_buf_linalg.mlir
../../build-ninja/tools/tutorial-opt --bufferization-to-llvm $PWD/sample_model_buf_linalg.mlir > $PWD/sample_model_llvm.mlir

###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir $PWD/sample_model_llvm.mlir > $PWD/sample_model_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj $PWD/sample_model_llvm_ir.ll

###  Compile  ###
gcc -c sample_model_main.cpp -o sample_model_main.o && gcc sample_model_main.o sample_model_llvm_ir.o -o a.out
gcc -c sample_model_main.cpp -o sample_model_main.o && gcc sample_model_main.o sample_model_llvm_ir.o -o a.out -L../../lib -lopenblas -lm

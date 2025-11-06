###  Pipeline to get from linalg to llvm dialect  ###
python import_flan_t5_small.py

# For Blas integration (Todo: Can be merged)
../../build-ninja/tools/tutorial-opt --linalg-to-bufferization $PWD/full_linalg.mlir > $PWD/full_buf_linalg.mlir
../../build-ninja/tools/tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm $PWD/full_buf_linalg.mlir > $PWD/full_llvm.mlir

###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir $PWD/full_llvm.mlir > $PWD/full_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj $PWD/full_llvm_ir.ll

###  Compile  ###
g++ -c flan_t5_small_model_main.cpp -o flan_t5_small_model_main.o && g++ flan_t5_small_model_main.o full_llvm_ir.o -lm -L../../externals/torch-mlir/build/lib -L../../lib -lmlir_c_runner_utils -Wl,-rpath=../../externals/torch-mlir/build/lib -lopenblas -o a.out

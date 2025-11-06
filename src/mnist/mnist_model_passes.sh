###  Pipeline to get from linalg to llvm dialect  ###
python torch_mlir_lowering_mnist_model.py

../../build-ninja/tools/tutorial-opt --linalg-to-llvm $PWD/mnist_model_linalg.mlir > $PWD/mnist_model_llvm.mlir

###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir $PWD/mnist_model_llvm.mlir > $PWD/mnist_model_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj $PWD/mnist_model_llvm_ir.ll

###  Compile  ###
#gcc -c sample_model_main.cpp -o sample_model_main.o && gcc sample_model_main.o sample_model_llvm_ir.o -o a.out -lm

gcc -c mnist_model_main.cpp -o mnist_model_main.o && gcc mnist_model_main.o mnist_model_llvm_ir.o -L../../externals/torch-mlir/build/lib -lmlir_c_runner_utils -Wl,-rpath=../../externals/torch-mlir/build/lib -o a.out
gcc -c mnist_model_main.cpp -o mnist_model_main.o && gcc mnist_model_main.o mnist_model_llvm_ir.o python/mnist/mnist_model_passes.sh
gcc -O3 -c mnist_model_benchmark.cpp -o bench.o && gcc bench.o mnist_model_llvm_ir.o -L../../externals/torch-mlir/build/lib -L../../lib -lmlir_c_runner_utils -Wl,-rpath=../../externals/torch-mlir/build/lib -lopenblas -o bench.out
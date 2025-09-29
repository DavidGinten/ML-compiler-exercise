###  Pipeline to get from linalg to llvm dialect  ###
python torch_mlir_lowering_resnet18_model.py

# torch-mlir: Convert from torch dialect to linalg dialect
../../externals/torch-mlir/build/bin/torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline $PWD/resnet18_model_torch.mlir > $PWD/resnet18_model_linalg.mlir

#mlir-opt --canonicalize --convert-elementwise-to-linalg --convert-tensor-to-linalg --one-shot-bufferize=bufferize-function-boundaries --buffer-deallocation-pipeline --convert-linalg-to-loops --expand-strided-metadata --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts --convert-func-to-llvm --canonicalize --sccp --cse --symbol-dce $PWD/python/sample_model_linalg.mlir
# or
../../build-ninja/tools/tutorial-opt --linalg-to-llvm $PWD/resnet18_model_linalg.mlir > $PWD/resnet18_model_llvm.mlir

# For Blas integration (Todo: Can be merged)
../../build-ninja/tools/tutorial-opt --linalg-to-bufferization $PWD/resnet18_model_linalg.mlir > $PWD/resnet18_model_buf_linalg.mlir
../../build-ninja/tools/tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm $PWD/resnet18_model_buf_linalg.mlir > $PWD/resnet18_model_llvm.mlir

###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir $PWD/resnet18_model_llvm.mlir > $PWD/resnet18_model_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj $PWD/resnet18_model_llvm_ir.ll

###  Compile  ###
#gcc -c resnet50_model_main.cpp -o resnet50_model_main.o && gcc resnet50_model_main.o resnet50_model_llvm_ir.o -o a.out -lm
gcc -c resnet18_model_main.cpp -o resnet18_model_main.o && gcc resnet18_model_main.o resnet18_model_llvm_ir.o -L../../externals/torch-mlir/build/lib -L../../lib -lmlir_c_runner_utils -Wl,-rpath=../../externals/torch-mlir/build/lib -lopenblas -o a.out

# Benchmark
gcc -O3 -c resnet50_model_benchmark.cpp -o bench.o && gcc bench.o resnet50_model_llvm_ir.o -o bench.out -L../../lib -lopenblas -lm
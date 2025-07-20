###  Pipeline to get from linalg to llvm dialect  ###
python torch_mlir_lowering_cnn_model.py

#mlir-opt --canonicalize --convert-elementwise-to-linalg --convert-tensor-to-linalg --one-shot-bufferize=bufferize-function-boundaries --buffer-deallocation-pipeline --convert-linalg-to-loops --expand-strided-metadata --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts --convert-func-to-llvm --canonicalize --sccp --cse --symbol-dce $PWD/python/sample_model_linalg.mlir
# or
../../build-ninja/tools/tutorial-opt --linalg-to-llvm $PWD/cnn_model_linalg.mlir > $PWD/cnn_model_llvm.mlir


###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir $PWD/cnn_model_llvm.mlir > $PWD/cnn_model_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj $PWD/cnn_model_llvm_ir.ll

###  Compile  ###
gcc -c cnn_model_main.cpp -o cnn_model_main.o && gcc cnn_model_main.o cnn_model_llvm_ir.o -o a.out -lm
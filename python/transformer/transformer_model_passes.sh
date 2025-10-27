###  Pipeline to get from linalg to llvm dialect  ###
python torch_mlir_lowering_resnet50_model.py

# Convert from torch to linalg on tensors
../../externals/torch-mlir/build/bin/torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline $PWD/transformer_model_torch.mlir > $PWD/transformer_model_linalg.mlir

#mlir-opt --canonicalize --convert-elementwise-to-linalg --convert-tensor-to-linalg --one-shot-bufferize=bufferize-function-boundaries --buffer-deallocation-pipeline --convert-linalg-to-loops --expand-strided-metadata --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts --convert-func-to-llvm --canonicalize --sccp --cse --symbol-dce $PWD/python/sample_model_linalg.mlir
# or
#../../build-ninja/tools/tutorial-opt --linalg-to-llvm $PWD/resnet50_model_linalg.mlir > $PWD/resnet50_model_llvm.mlir

# For Blas integration (Todo: Can be merged)
../../build-ninja/tools/tutorial-opt --linalg-to-bufferization $PWD/flan_t5_small_linalg.mlir > $PWD/flan_t5_small_buf_linalg.mlir
../../build-ninja/tools/tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm $PWD/flan_t5_small_buf_linalg.mlir > $PWD/flan_t5_small_llvm.mlir

###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir $PWD/t5_encoder_llvm.mlir > $PWD/t5_encoder_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj $PWD/flan_t5_small_llvm_ir.ll

###  Compile  ###
gcc -c flan_t5_small_main.cpp -o flan_t5_small_main.o && gcc flan_t5_small_main.o flan_t5_small_llvm_ir.o -o a.out -lm
gcc -c flan_t5_small_main.cpp -o flan_t5_small_main.o && gcc flan_t5_small_main.o flan_t5_small_llvm_ir.o -o a.out -L../../lib -lopenblas -lm

gcc -c flan_t5_small_main.cpp -o flan_t5_small_main.o && gcc flan_t5_small_main.o t5_encoder_llvm_ir.o t5_decoder_step_llvm_ir.o -lm -L../../externals/torch-mlir/build/lib -L../../lib -lmlir_c_runner_utils -Wl,-rpath=../../externals/torch-mlir/build/lib -lopenblas -o a.out
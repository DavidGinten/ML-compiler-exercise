# ML compiler exercise

This is a experimenting MLIR pipeline to lower ML models from PyTorch to x86. It is about to become an exercise 
for a new lecture at RWTH Aachen.

# Getting started
Instructions are partially RWTH cluster specific.

Initalize the submodules (torch-mlir, llvm-project)
`git submodule update --init --recursive`

Load PYTHON 3.12 before you start (or load by default in ~/zshrc)
`module load PYTHON/3.12`

Set-up a python virtual environment
```
python3 -m venv venv_torch_mlir
source venv_torch_mlir/bin/activate
pip install --upgrade pip
```

Install latest requirements from torch-mlir
`python -m pip install -r requirements.txt -r torchvision-requirements.txt`

Configuration for building torch-mlir with llvm in-tree.
Use ccache if available.
Also refer to the torch-mlir [development guide](https://github.com/llvm/torch-mlir/blob/main/docs/development.md).
You can also install the MLIR Python bindings from prebuilt wheels (See requirements.txt)
```
cmake -GNinja -Bbuild \
  `# Enables "--debug" and "--debug-only" flags for the "torch-mlir-opt" tool` \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DPython_FIND_VIRTUALENV=ONLY \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  `# For building LLVM "in-tree"` \
  externals/llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
  -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON
```

With GPU
```
cmake -GNinja -Bbuild \
  `# Enables "--debug" and "--debug-only" flags for the "torch-mlir-opt" tool` \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DPython_FIND_VIRTUALENV=ONLY \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  `# For building LLVM "in-tree"` \
  externals/llvm-project/llvm \
  -DCUDACXX=/path/to/nvcc \
  -DCUDA_PATH=/path/to/cuda \
  -DCMAKE_CUDA_ARCHITECTURES="90" \ # Use your CUDA GPU Compute Capability
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CUDA_COMPILER=/path/to/nvcc \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
  -DLLVM_CCACHE_BUILD=OFF  \
  -DMLIR_ENABLE_CUDA_RUNNER=ON \
  -DMLIR_ENABLE_CUDA_CONVERSIONS=ON \
  -DMLIR_ENABLE_NVPTXCOMPILER=ON \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
  -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON \
```

Build (and inital testing)
`cmake --build build --target check-torch-mlir --target check-torch_mlir-python`

Or use Ninja directly
`ninja -C build`

Tests: `ninja check-torch-mlir check-torch-mlir-python`

In venv_torch_mlir/bin/activate add the following (adapt the path if necessary):
```
# Add torch-mlir-opt to PATH
export PATH="/home/ab123456/ml-compiler-exercise/externals/torch-mlir/build/bin/:$PATH"

# Add MLIR Python bindings and Setup Python Environment to export the built Python packages
export PYTHONPATH=/home/ab123456/ml-compiler-exercise/externals/torch-mlir/build/tools/mlir/python_packages/mlir_core:/home/ab123456/ml-compiler-exercise/externals/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir:/home/ab123456/ml-compiler-exercise/externals/torch-mlir/test/python/fx_importer
```
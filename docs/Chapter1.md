# Chapter 1: Introduction and project setup
Credits: The blog article by [Jeremy Kun](https://www.jeremykun.com/2023/08/10/mlir-getting-started/) helped me a lot to build this pipeline. Especially [this](https://www.jeremykun.com/2023/11/01/mlir-lowering-through-llvm/) article to lower MLIR dialects.

For a general introduction I highly recommend Stephan Diehl's [MLIR GPU lowering posts](https://www.stephendiehl.com/tags/mlir/)
## Introduction

MLIR is more and more used in the compilation process of machine learning models. As opposed to other compilers, IREE in its compilation process has multiple phases in which it can represent its IR in different stages. These stages are different representations of the model with its own set of instructions and referred to as *MLIR dialects*. A classification of dialects can be found [here](https://youtu.be/hIt6J1_E21c?t=800). This multi-phase approach allows to have more level of abstraction and with that better ways to target devices and is a lot harder to achieve in normal two phase compiler. Different optimizations  can be applied on different abstarction levels where they are best suited. Usually, you have some program that inputs your models from the its defining framework (e.g. PyTorch, TensorFlow, Onnx) to a high level MLIR dialect, e.g. the linalg dialect. For example, torch-mlir reads the PyTorch model's graph representation and creates a representation in the torch dialect (can also be linalg). Then the power of MLIR comes in. Many passes exit to incrementaly lower the dialects, form high-level ones to more assembly like dialects (so called exit dialect, e.g. llvm). Thereby, many optimizations can also be performed on the corresponding dialects. One can also define its own dialects, passes, etc. Different types of passes exist. For example, transform passes transform code within a dialect, conversion passes convert code between dialects, analysis passes include anaylsis passes like data-flow analysis etc.    
Once arrived at an exit dialect, we need to translate to llvm ir, thereby leaving the MLIR space. From here, object code an be generated for a specific target. Then the model can be called from example C++ by a regular function call.   

## A state-of-the-art ML compiler: IREE
A large and often used state-of-the-art ML-compiler that is MLIR-based is [IREE](https://iree.dev/). It consits of the end-to-end compiler and runtime. The created IR is optimized for real-time machine learning inference on edge/mobile devices running on various hardware accelerators. The first step is to build the model in the framework of choice, e.g. Tensorflow, PyTorch. The source model received out of the frontend is then feed into the IREE compiler. The compiler builds different shared library-like modules called artifacts with the executable binary for target backends (the execution logic) and the partitioned host program (the scheduling logic) among other small programs. The last component in the IREE workflow is the IREE runtime. It dynamically links and loads modules into isolated contexts. With an Hardware abstraction laxer (HAL) it can easily make use of hardware backends and has multiple API's help for ease of use. 

The following describtion of [IREE's project architecture](https://iree.dev/#project-architecture) depicted below. First, the created model is translated into an MLIR representation by the framework. It takes a snapshot of the structure and data constituting the model and translates it into a MLIR dialect which IREE accepts as input. For example, the dialect for TensorFlow Lite is Tosa. $linalg$ and $arith$ depicted in the picture are also some valid input dialects. Those dialects are then feed into the compiler. (For example, the IREE importer for PyTorch models is [iree-turbine](https://github.com/iree-org/iree-turbine))The IREE compiler stack has 3 stages helping in lowering the operations and each being a Intermediate Representation (or MLIR dialect). The first one is the flow dialect which captures high-level model structure and divides computations into dispatchable regions, which can later be optimized and scheduled. In the picture the partitioned workload can be seen. The next stage enables the efficient execution of machine learning models by modeling workloads as streams of data and operations that can be scheduled and executed asynchronously and parallel. It makes these workloads well-suited for high-performance machine learning and other compute-heavy applications and is crucial for optimizing the execution of large, complex workloads where managing computation and data flow efficiently is essential. The Hardware Abstraction Layer (HAL) is the third layer in the IREE compiler managing buffers and defines how operations should be executed on different devices. IREE supports multiple target backends for different hardware architectures. These backends lower the intermediate representation into code that is optimized for a specific hardware target. For example, the LLVM backend targets CPUs, and is optimized for scalar and vectorized execution on a variety of architectures, e.g. x86, ARM. For targeting GPUs, e.g., Vulkan, OpenCL, the SPIR-V backend is used which lowers the IR into SPIR-V, a widely used intermediate representation for GPU execution.
As mentioned in the overview, the compiler issues IREE modules which then are forwarded into the small IREE runtime. The runtime efficiently executes compiled models on different hardware and consitits of several Plugins, a HAL and a IREE Virtual Machine (VM). Furthermore, API bindings are provided for IREE's compiler and runtime to be used in programming. The number of languages is under development and therefore still growing. To run individual compiler passes, translations, and other transformations step by step IREE privides tools for these tasks. Some examples are given in the picture.  

![The IREE project architecture](https://iree.dev/assets/images/iree_architecture_dark.svg#gh-dark-mode-only)

## What this tutuorial is about
In this tutorial, we will build our own MLIR pipeline to lower some real ML-models. Specifically, we take the models form PyTorch, import them with [torch-mlir](https://github.com/llvm/torch-mlir), and then select multiple passes to lower the model to the MLIR llvm dialect. We'll then export the model to x86 assembly code to be then called by C/C++. We will also write our own pass that converts matrix multiplications to OpenBLAS function calls to speed up the execution. 

## The project setup
The project setup can be found [here](link). It contains the following directories:
- **externals** includes torch-mlir, and with that LLVM/MLIR in tree
- **lib** to store our own passes
- **python** that imports the PyTorch models to MLIR
- **tools** contains our pass pipeline

Aside: Usually, you also have an include/ folder that holds the header files whereas lib/ holds the implementations files. lib/ also has subdirectories for the different kinds of passes we mentioned above (e.g. lib/Transform). But we put together the headers and implementations in lib/ as we have not many passes (currently only 1).

## Getting started 
First we need to initalize the submodules (torch-mlir, llvm-project) `git submodule update --init --recursive`

Set-up a python virtual environment to install the lastest requirements from torch-mlir
```
python3 -m venv venv_torch_mlir
source venv_torch_mlir/bin/activate
pip install --upgrade pip

python -m pip install -r requirements.txt -r torchvision-requirements.txt
```

We can then configure the built in torch-mlir (we built llvm/mlir in-tree)
```
cmake -GNinja -Bbuild \
  `# Enables "--debug" and "--debug-only" flags for the "torch-mlir-opt" tool` \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
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

Build (and inital testing) `cmake --build build --target check-torch-mlir //--target check-mlir --target check-torch_mlir-python`
Or use Ninja directly `ninja -C build check-torch-mlir`

In venv_torch_mlir/bin/activate add the following (adapt the path if necessary):
```
# Add torch-mlir-opt to PATH
export PATH="/home/ab123456/ml-compiler-exercise/externals/torch-mlir/build/bin/:$PATH"

# Add MLIR Python bindings and Setup Python Environment to export the built Python packages
export PYTHONPATH=/home/ab123456/ml-compiler-exercise/externals/torch-mlir/build/tools/mlir/python_packages/mlir_core:/home/ab123456/ml-compiler-exercise/externals/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir:/home/ab123456/ml-compiler-exercise/externals/torch-mlir/test/python/fx_importer
```

In the next chapter we will write simple, example PyTorch models and import them to torch-mlir to visit the model in MLIR torch dialect. 
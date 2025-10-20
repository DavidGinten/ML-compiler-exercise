#!/bin/sh

BUILD_SYSTEM="Ninja"
BUILD_DIR=./build-`echo ${BUILD_SYSTEM}| tr '[:upper:]' '[:lower:]'`

rm -rf $BUILD_DIR
mkdir $BUILD_DIR
pushd $BUILD_DIR

LLVM_BUILD_DIR=externals/torch-mlir/build
gcc -c cnn_model_main_test.cpp -o cnn_model_main_test.o && gcc cnn_model_main_test.o cnn_model_llvm_ir.o -o a.out -L../../lib -lopenblas -lm

popd

cmake --build $BUILD_DIR --target mlir-headers
cmake --build $BUILD_DIR --target mlir-doc
cmake --build $BUILD_DIR --target tutorial-opt
#cmake --build $BUILD_DIR --target check-mlir-tutorial

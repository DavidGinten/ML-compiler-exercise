// #include "mlir-c/Dialect/MemRef.h"
// #include "mlir/CAPI/Registration.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <cstdint>
#include <cstdio>
#include <stdio.h>

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void _mlir_ciface_cnn_model(MemRefDescriptor<float, 2> *output,
                            MemRefDescriptor<float, 4> *input);
}

int main(int argc, char *argv[]) {
  float inputData[32][1][28][28];
  float outputData[32][10];
  for (int k = 0; k < 32; k++) {
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        inputData[k][0][i][j] = 1.0;
      }
    }
  }
  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 10; j++) {
      outputData[i][j] = 0.0;
    }
  }
  int64_t offset = 0;
  int64_t sizes[4] = {32, 1, 28, 28};
  int64_t strides[4] = {784, 784, 28, 1}; // row-major layout

  int64_t sizes_output[2] = {32, 10};
  int64_t strides_output[2] = {10, 1}; // row-major layout

  MemRefDescriptor<float, 4> inputMemRef = {
      (float *)inputData,
      (float *)inputData,
      offset,
      {sizes[0], sizes[1], sizes[2], sizes[3]},
      {strides[0], strides[1], strides[2], strides[3]}};

  MemRefDescriptor<float, 2> outputMemRef = {
      (float *)outputData,
      (float *)outputData,
      offset,
      {sizes_output[0], sizes_output[1]},
      {strides_output[0], strides_output[1]}};

  // Call the model
  _mlir_ciface_cnn_model(&outputMemRef, &inputMemRef);


  float * output_new = (float *)outputMemRef.aligned;


  for (int64_t i = 0; i < sizes_output[0]; ++i) {
    for (int64_t j = 0; j < sizes_output[1]; ++j) {
      printf("%.4f ", output_new[i * strides_output[0] + j * strides_output[1]]);
    }
    printf("\n");
  }

  return 0;
}
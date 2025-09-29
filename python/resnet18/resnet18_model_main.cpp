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
void _mlir_ciface_resnet18(MemRefDescriptor<float, 2> *output,
                           MemRefDescriptor<float, 4> *input);
}

int main(int argc, char *argv[]) {
  float inputData[1][3][224][224];
  float outputData[1][1000];
  for (int k = 0; k < 3; k++) {
    for (int i = 0; i < 224; i++) {
      for (int j = 0; j < 224; j++) {
        inputData[0][k][i][j] = 1.0;
      }
    }
  }
  for (int j = 0; j < 1000; j++) {
    outputData[0][j] = 0.0;
  }

  // Create MemRef descriptors
  int64_t offset = 0;
  int64_t input_sizes[4] = {1, 3, 224, 224};
  int64_t input_strides[4] = {3*224*224, 224*224, 224, 1}; // row-major layout

  int64_t output_size[2] = {1, 1000};
  int64_t output_strides[2] = {1000, 1}; // row-major layout

  MemRefDescriptor<float, 4> inputMemRef = {
      (float *)inputData,
      (float *)inputData,
      offset,
      {input_sizes[0], input_sizes[1], input_sizes[2], input_sizes[3]},
      {input_strides[0], input_strides[1], input_strides[2], input_strides[3]}};

  MemRefDescriptor<float, 2> outputMemRef = {
      (float *)outputData,
      (float *)outputData,
      offset,
      {output_size[0], output_size[1]},
      {output_strides[0], output_strides[1]}};

  // Call the model
  _mlir_ciface_resnet18(&outputMemRef, &inputMemRef);

  float *output_new = (float *)outputMemRef.aligned;

  int count = 0;
  for (int64_t i = 0; i < output_size[1]; ++i) {
    printf("%.5f ", output_new[i]);
    count++;
  }
  printf("\nCount = %d\n", count);

  return 0;
}
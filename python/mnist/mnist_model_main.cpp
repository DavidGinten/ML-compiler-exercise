#include <stdio.h>

#include <cstdint>
#include <cstdio>

extern "C" {
    typedef struct {
        void* allocated;
        void* aligned;
        int64_t offset;
        int64_t sizes[2];
        int64_t strides[2];
    } MemRef2D;

    MemRef2D mnist_model(
        void* allocated,
        void* aligned,
        int64_t offset,
        int64_t size0,
        int64_t size1,
        int64_t size2,
        int64_t stride0,
        int64_t stride1,
        int64_t stride2
    );
}

int main(int argc, char *argv[]) {
  float inputData[1][28][28];

    for(int ii=0; ii<28; ii++){
        for(int jj=0; jj<28; jj++){
            inputData[0][ii][jj] = 1.0;
        }
    }
  
  // No offset for simple cases
  int64_t offset = 0;
  int64_t sizes[3]= {1, 28, 28};
  int64_t strides[3] = {1, 28, 1};  // row-major layout
  
  // Call the model
  MemRef2D result = mnist_model(
      inputData,       // allocated
      inputData,         // aligned
      offset, 
      sizes[0], sizes[1], sizes[2],
      strides[0], strides[1], strides[2]);

  // Access output buffer
  float* output = (float*)(result.aligned);
  
  //printf("Output tensor (shape: %ld x %ld):\n", result.sizes[0], result.sizes[1]);
  //printf("result.strides[0]: %ld\n", result.strides[0]);
  //printf("result.strides[1]: %ld\n", result.strides[1]);
  for (int64_t i = 0; i < result.sizes[0]; ++i) {
      for (int64_t j = 0; j < result.sizes[1]; ++j) {
          printf("%.4f ", output[i * result.strides[0] + j * result.strides[1]]);
      }
      printf("\n");
  }

  return 0;
}
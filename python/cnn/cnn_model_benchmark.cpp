#include <stdio.h>

#include <cstdint>
#include <cstdio>
#include <time.h>

extern "C" {
    typedef struct {
        void* allocated;
        void* aligned;
        int64_t offset;
        int64_t sizes[2];
        int64_t strides[2];
    } MemRef2D;

    MemRef2D cnn_model(
        void* allocated,
        void* aligned,
        int64_t offset,
        int64_t size0,
        int64_t size1,
        int64_t size2,
        int64_t size3,
        int64_t stride0,
        int64_t stride1,
        int64_t stride2,
        int64_t stride3
    );
}

int main(int argc, char *argv[]) {
    float inputData[32][1][28][28];

    for(int k=0; k<32;k++){
        for(int i=0; i<28; i++){
            for(int j=0; j<28; j++){
                inputData[k][0][i][j] = 1.0; 
            }
        }
    }   

    // No offset for simple cases
    int64_t offset = 0;
    int64_t sizes[4] = {32, 1, 28, 28};
    int64_t strides[4] = {784,784, 28, 1};  // row-major layout

    const int runs = 100;
    struct timespec start, end;
    
    // Warm-up
    for (int i = 0; i < 10; ++i) {
        // Call the model
        cnn_model(inputData, inputData, offset, 
            sizes[0], sizes[1], sizes[2], sizes[3],
            strides[0], strides[1], strides[2], strides[3]);
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < runs; ++i) {
        // Call the model
        cnn_model(inputData, inputData, offset, 
            sizes[0], sizes[1], sizes[2], sizes[3],
            strides[0], strides[1], strides[2], strides[3]);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                 ((end.tv_nsec - start.tv_nsec) / 1e9);

    printf("Avg inference time: %.6f sec\n", elapsed / runs);

    return 0;
}
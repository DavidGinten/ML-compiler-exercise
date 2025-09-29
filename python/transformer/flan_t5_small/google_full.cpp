#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <vector>

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void _mlir_ciface_transformer_model(
    MemRefDescriptor<float, 3> *output, MemRefDescriptor<int64_t, 2> *input_ids,
    MemRefDescriptor<int64_t, 2> *attention_mask,
    MemRefDescriptor<int64_t, 2> *decoder_input_ids);
}

int main(int argc, char *argv[]) {
  int64_t input_ids[1][11] = {{13959, 1566, 12, 2968, 10, 571, 625, 33, 25, 58,
                               1}}; // Example input IDs
  int64_t attention_mask[1][11] = {
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}; // Example attention mask

  // Create MemRef descriptors
  int64_t offset = 0;

  MemRefDescriptor<int64_t, 2> input_ids_MemRef = {
      (int64_t *)input_ids, (int64_t *)input_ids, offset, {1, 11}, {11, 1}};

  MemRefDescriptor<int64_t, 2> attention_mask_MemRef = {
      (int64_t *)attention_mask,
      (int64_t *)attention_mask,
      offset,
      {1, 11},
      {11, 1}};
  int max_len = 10;
  std::vector<int64_t> decoder_input_ids = {0};
  std::vector<float> outputData(32128);

  for (int i = 0; i < max_len; i++) {
    int64_t current_len = static_cast<int64_t>(decoder_input_ids.size());

    // Allocate output buffer INSIDE loop for current decoder length

    // Decoder input MemRef
    MemRefDescriptor<int64_t, 2> decoder_input_ids_MemRef = {
        decoder_input_ids.data(),
        decoder_input_ids.data(),
        0,
        {1, current_len},
        {current_len, 1}};

    // Decoder output MemRef
    MemRefDescriptor<float, 3> outputMemRef = {outputData.data(),
                                               outputData.data(),
                                               0,
                                               {1, 1, 32128},
                                               {32128, 32128, 1}};

    // Run model
    _mlir_ciface_transformer_model(&outputMemRef, &input_ids_MemRef,
                                   &attention_mask_MemRef,
                                   &decoder_input_ids_MemRef);

    // Access through the aligned pointer from the descriptor
    float *output = outputMemRef.allocated;
    // float *last_logits = output + (current_len - 1) * 32128;

    int index = std::distance(
        output, std::max_element(output, output + 32128));

    printf("Step %d, Index = %d\n", i, index);
    decoder_input_ids.push_back(index);
  }

  return 0;
}
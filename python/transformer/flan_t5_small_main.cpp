#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>
#include <iostream>

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void _mlir_ciface_t5_encoder(MemRefDescriptor<float, 3> *output,
                             MemRefDescriptor<int64_t, 2> *input_ids,
                             MemRefDescriptor<int64_t, 2> *attention_mask);
void _mlir_ciface_t5_decoder_step(
    MemRefDescriptor<float, 3> *output,
    MemRefDescriptor<int64_t, 2> *last_token_id,
    MemRefDescriptor<float, 3> *encoder_hidden_states,
    MemRefDescriptor<int64_t, 2> *attention_mask);
}

int main(int argc, char *argv[]) {
  int64_t input_ids[1][11] = {{13959, 1566, 12, 2968, 10, 571, 625, 33, 25, 58,
                               1}}; // Example input IDs
  int64_t attention_mask[1][11] = {
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}; // Example attention mask

  int64_t last_token_id[1][1] = {{3}}; // Example decoder start token ID

  float outputData[1][11][512];
  float real_outputData[1][1][32128];

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

  MemRefDescriptor<int64_t, 2> last_token_id_MemRef = {(int64_t *)last_token_id,
                                                       (int64_t *)last_token_id,
                                                       offset,
                                                       {1, 1},
                                                       {1, 1}};

  MemRefDescriptor<float, 3> encoder_hidden_states_MemRef = {
      (float *)outputData,
      (float *)outputData,
      offset,
      {1, 11, 512},
      {11 * 512, 512, 1}};

  MemRefDescriptor<float, 3> output_MemRef = {(float *)real_outputData,
                                              (float *)real_outputData,
                                              offset,
                                              {1, 1, 32128},
                                              {32128, 32128, 1}};
  // Call the model
  _mlir_ciface_t5_encoder(&encoder_hidden_states_MemRef, &input_ids_MemRef,
                          &attention_mask_MemRef);

  _mlir_ciface_t5_decoder_step(&output_MemRef, &last_token_id_MemRef,
                               &encoder_hidden_states_MemRef,
                               &attention_mask_MemRef);

  float *output = (float *)output_MemRef.aligned;

  /*
  printf("Output shape: [%ld, %ld, %ld]\n", output_MemRef.sizes[0],
         output_MemRef.sizes[1], output_MemRef.sizes[2]);

  encoder_hidden_states_MemRef.sizes[0], encoder_hidden_states_MemRef.sizes[1],
  encoder_hidden_states_MemRef.sizes[2]);
  */
  
  std::vector<float> output_vector(
      output, output + 32128); // Flatten the 3D array to 1D vector

  // printf("Value 0: %f\n", *output+10);
  std::vector<int> indices(output_vector.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return output_vector[a] > output_vector[b]; });

  for (int i = 0; i < 5; i++) {
    std::cout << "Value: " << output_vector[indices[i]] << " at index " << indices[i]
              << "\n";
  }

  return 0;
}
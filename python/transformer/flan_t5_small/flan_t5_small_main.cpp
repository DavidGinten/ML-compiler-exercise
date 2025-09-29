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

  int64_t decoder_input_ids[1][1] = {{0}}; // Example decoder start token ID

  float outputData[1][1][32128];

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

  MemRefDescriptor<int64_t, 2> decoder_input_ids_MemRef = {
      (int64_t *)decoder_input_ids,
      (int64_t *)decoder_input_ids,
      offset,
      {1, 1},
      {1, 1}};

  MemRefDescriptor<float, 3> outputMemRef = {(float *)outputData,
                                             (float *)outputData,
                                             offset,
                                             {1, 1, 32128},
                                             {32128, 32128, 1}};

  // Call the model
  _mlir_ciface_transformer_model(&outputMemRef, &input_ids_MemRef,
                                 &attention_mask_MemRef,
                                 &decoder_input_ids_MemRef);

  float *output = (float *)outputMemRef.aligned;
  int index = std::distance(output, std::max_element(output, output + 32128));
  std::vector<float> output_vector(
      output, output + 32128); // Flatten the 3D array to 1D vector

  // printf("Value 0: %f\n", *output+10);
  std::vector<int> indices(output_vector.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return output_vector[a] > output_vector[b]; });

  for (int i = 0; i < 5; i++) {
    std::cout << "Value: " << output_vector[indices[i]] << " at index "
              << indices[i] << "\n";
  }
  std::cout << "\n";
  int64_t decoder_input_ids_2[1][2] = {
      {0, 2739}}; // Example decoder start token ID
  float outputData_2[1][2][32128];

  MemRefDescriptor<int64_t, 2> decoder_input_ids_2_MemRef = {
      (int64_t *)decoder_input_ids_2,
      (int64_t *)decoder_input_ids_2,
      offset,
      {1, 2},
      {2, 1}};

  MemRefDescriptor<float, 3> output_2_MemRef = {(float *)outputData_2,
                                                (float *)outputData_2,
                                                offset,
                                                {1, 2, 32128},
                                                {2 * 32128, 32128, 1}};
  _mlir_ciface_transformer_model(&output_2_MemRef, &input_ids_MemRef,
                                 &attention_mask_MemRef,
                                 &decoder_input_ids_2_MemRef);

  float *output_2 = (float *)output_2_MemRef.aligned;
  std::vector<float> output_vector_2(
      output_2, output_2 + 32128); // Flatten the 3D array to 1D vector

  // printf("Value 0: %f\n", *output+10);
  std::vector<int> indices_2(output_vector_2.size());
  std::iota(indices_2.begin(), indices_2.end(), 0);

  std::sort(indices_2.begin(), indices_2.end(), [&](int a, int b) {
    return output_vector_2[a] > output_vector_2[b];
  });

  for (int i = 0; i < 5; i++) {
    std::cout << "Value: " << output_vector_2[indices_2[i]] << " at index "
              << indices_2[i] << "\n";
  }
  std::cout << "\n";

  int64_t decoder_input_ids_3[1][3] = {
      {0, 2739, 3}}; // Example decoder start token ID
  float outputData_3[1][3][32128];

  MemRefDescriptor<int64_t, 2> decoder_input_ids_3_MemRef = {
      (int64_t *)decoder_input_ids_3,
      (int64_t *)decoder_input_ids_3,
      offset,
      {1, 3},
      {3, 1}};

  MemRefDescriptor<float, 3> output_3_MemRef = {(float *)outputData_3,
                                                (float *)outputData_3,
                                                offset,
                                                {1, 3, 32128},
                                                {3 * 32128, 32128, 1}};
  _mlir_ciface_transformer_model(&output_3_MemRef, &input_ids_MemRef,
                                 &attention_mask_MemRef,
                                 &decoder_input_ids_3_MemRef);

  float *output_3 = (float *)output_3_MemRef.aligned;
  std::vector<float> output_vector_3(
      output_3, output_3 + 32128); // Flatten the 3D array to 1D vector

  // printf("Value 0: %f\n", *output+10);
  std::vector<int> indices_3(output_vector_3.size());
  std::iota(indices_3.begin(), indices_3.end(), 0);

  std::sort(indices_3.begin(), indices_3.end(), [&](int a, int b) {
    return output_vector_3[a] > output_vector_3[b];
  });

  for (int i = 0; i < 5; i++) {
    std::cout << "Value: " << output_vector_3[indices_3[i]] << " at index "
              << indices_3[i] << "\n";
  }

  printf("\nIndex = %u\n", index);
  // printf("\nIndex = %u\n", index_2);

  return 0;
}
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cmath>

template <typename T, int N> struct MemRefDescriptor {
    T *allocated;
    T *aligned;
    int64_t offset;
    int64_t sizes[N];
    int64_t strides[N];
};

extern "C" {
    void _mlir_ciface_transformer_model(
        MemRefDescriptor<float, 2> *output,
        MemRefDescriptor<int64_t, 2> *input_ids,
        MemRefDescriptor<int64_t, 2> *attention_mask,
        MemRefDescriptor<int64_t, 2> *decoder_input_ids);
}

int main(int argc, char *argv[]) {
    // Example input IDs ("translate English to German: How are you?")
    int64_t input_ids[1][10] = {{13959, 1566, 12, 2968, 10, 571, 33, 25, 58, 1}};
    int64_t attention_mask[1][10] = {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
    
    int64_t offset = 0;
    MemRefDescriptor<int64_t, 2> input_ids_MemRef = {
        (int64_t *)input_ids, (int64_t *)input_ids, offset, {1, 10}, {10, 1}
    };
    MemRefDescriptor<int64_t, 2> attention_mask_MemRef = {
        (int64_t *)attention_mask, (int64_t *)attention_mask, offset, {1, 10}, {10, 1}
    };
    
    std::vector<int64_t> decoder_input_ids = {0};
    std::vector<float> outputData(32128);
    
    for (int i = 0; i < 10; i++) {
        int64_t current_len = static_cast<int64_t>(decoder_input_ids.size());
        
        std::fill(outputData.begin(), outputData.end(), -999.0f);
        
        printf("\nStep %d, decoder_len = %ld\n", i, current_len);
        printf("  Decoder tokens: [");
        for (size_t j = 0; j < decoder_input_ids.size(); j++) {
            printf("%ld%s", decoder_input_ids[j], j < decoder_input_ids.size()-1 ? ", " : "");
        }
        printf("]\n");
        
        MemRefDescriptor<int64_t, 2> decoder_input_ids_MemRef = {
            decoder_input_ids.data(),
            decoder_input_ids.data(),
            0,
            {1, current_len},
            {current_len, 1}
        };
        
        MemRefDescriptor<float, 2> outputMemRef = {
            outputData.data(),
            outputData.data(),
            0,
            {1, 32128},
            {32128, 1}
        };
        
        printf("  Calling model...\n");
        _mlir_ciface_transformer_model(&outputMemRef, &input_ids_MemRef,
                                       &attention_mask_MemRef,
                                       &decoder_input_ids_MemRef);
        
        float *output = outputMemRef.aligned;
        
        // Check if output was actually written
        bool has_valid_output = false;
        for (int j = 0; j < 100; j++) {
            if (output[j] != -999.0f) {
                has_valid_output = true;
                break;
            }
        }
        
        if (!has_valid_output) {
            printf("  ERROR: Output buffer not written!\n");
            break;
        }
        
        // Find top 5 predictions
        std::vector<std::pair<float, int>> scores;
        for (int j = 0; j < 32128; j++) {
            if (!std::isnan(output[j]) && !std::isinf(output[j])) {
                scores.push_back({output[j], j});
            }
        }
        
        if (scores.empty()) {
            printf("  ERROR: All outputs are NaN or Inf!\n");
            break;
        }
        
        std::partial_sort(scores.begin(), scores.begin() + std::min(5, (int)scores.size()), 
                         scores.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        printf("  Top 5 predictions:\n");
        for (int j = 0; j < std::min(5, (int)scores.size()); j++) {
            printf("    Token %d: %.4f\n", scores[j].second, scores[j].first);
        }
        
        int index = scores[0].second;
        printf("  Selected: %d\n", index);
        
        if (index == 1) {
            printf("  EOS token generated, stopping.\n");
            break;
        }
        
        decoder_input_ids.push_back(index);
    }
    
    printf("\nFinal sequence: [");
    for (size_t i = 0; i < decoder_input_ids.size(); i++) {
        printf("%ld%s", decoder_input_ids[i], i < decoder_input_ids.size()-1 ? ", " : "");
    }
    printf("]\n");
    
    return 0;
}
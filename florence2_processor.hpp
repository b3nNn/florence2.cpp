#ifndef FLORENCE2_PROCESSOR_HPP
#define FLORENCE2_PROCESSOR_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "bart_tokenizer_fast.hpp"
#include "clip_image_processor.hpp"

namespace Florence2Processor {

    struct PastKeyValueLayer {
        // Define as needed
    };

    struct ModelOutput {
        std::vector<float> logits;
        std::vector<PastKeyValueLayer> past_key_values;
    };

    class Florence2Processor {
    public:
        Florence2Processor(const std::string& gguf_path, const std::string& tokenizer_json_path, const std::string& image_config_path, const std::string& mapping_path);
        ~Florence2Processor();

        ModelOutput process(const std::string& text, const cv::Mat& image, const std::vector<int64_t>& decoder_input_ids,
                            const std::vector<PastKeyValueLayer>& past_key_values);
        std::string decode(const std::vector<uint32_t>& token_ids, bool skip_special_tokens);
        std::string generate(const std::string& text, const cv::Mat& image, int max_length);

    private:
        ggml_context* ctx;
        ggml_context* gguf_ctx;
        ggml_gallocr_t galloc;
        ggml_cgraph* graph;
        ggml_tensor* pixel_values_prealloc;
        ggml_tensor* conv0_output;
        ggml_tensor* conv0_bias;
        ggml_tensor* norm0_prealloc;
        ggml_tensor* conv1_output;
        ggml_tensor* conv1_bias;
        ggml_tensor* norm1_prealloc;
        ggml_tensor* conv2_output;
        ggml_tensor* conv2_bias;
        ggml_tensor* norm2_prealloc;
        ggml_tensor* input_ids_tensor_prealloc;
        ggml_tensor* decoder_input_ids_tensor_prealloc;

        std::map<std::string, std::string> tensor_mapping;
        std::map<std::string, std::string> tensor_full_names;
        std::unordered_map<std::string, ggml_tensor*> tensors;
        BartTokenizerFast tokenizer;
        CLIPImageProcessor image_processor;

        int vocab_size;
        int t_dec_layers;
        int t_enc_layers;
        int t_heads;
        int t_hidden;
        int t_ffn_dim;
        int v_img_size;

        void initialize_ggml_context(const std::string& gguf_path);
        void reset_graph_allocator();
        ggml_tensor* load_tensor(const std::string& name);
        int get_metadata_int(gguf_context* ctx, const std::string& key);

        // New helper for manual matrix multiplication
        ggml_tensor* manual_mul_mat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

        // New function for image processing
        ggml_tensor* process_image(const cv::Mat& image);
    };

} // namespace Florence2Processor

#endif // FLORENCE2_PROCESSOR_HPP
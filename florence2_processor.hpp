#ifndef FLORENCE2_PROCESSOR_HPP
#define FLORENCE2_PROCESSOR_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ggml.h"
#include "gguf.h"  // From llama.cpp for GGUF parsing
#include "bart_tokenizer_fast.hpp"
#include "clip_image_processor.hpp"

namespace Florence2Processor {

    struct PastKeyValueLayer {
        std::vector<float> self_key;
        std::vector<float> self_value;
        std::vector<float> cross_key;
        std::vector<float> cross_value;
    };

    struct ModelOutput {
        std::vector<float> logits;
        std::vector<PastKeyValueLayer> past_key_values;
    };

    class Florence2Processor {
    public:
        Florence2Processor(const std::string& gguf_path, const std::string& tokenizer_json_path, const std::string& image_config_path);
        ~Florence2Processor();

        ModelOutput process(const std::string& text, const cv::Mat& image, const std::vector<int64_t>& decoder_input_ids,
                            const std::vector<PastKeyValueLayer>& past_key_values);
        std::string decode(const std::vector<uint32_t>& token_ids, bool skip_special_tokens);
        std::string generate(const std::string& text, const cv::Mat& image, int max_length);

    private:
        ggml_context* ctx;
        gguf_context* gguf_ctx;
        std::unordered_map<std::string, ggml_tensor*> tensors;
        BartTokenizerFast tokenizer;
        CLIPImageProcessor image_processor;

        // Model parameters from GGUF metadata
        int vocab_size;
        int t_dec_layers;
        int t_enc_layers;
        int t_heads;
        int t_hidden;
        int t_ffn_dim;
        int v_img_size;

        // Helper functions
        ggml_tensor* load_tensor(const std::string& name);
        void initialize_ggml_context(const std::string& gguf_path);
        int get_metadata_int(gguf_context* ctx, const std::string& key);
    };

} // namespace Florence2Processor

#endif // FLORENCE2_PROCESSOR_HPP
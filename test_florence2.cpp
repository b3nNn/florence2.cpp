#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <cassert>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-cpu.h>
#include <gguf.h>
#include <opencv2/opencv.hpp>
#include "bart_tokenizer_fast.hpp"
#include "clip_image_processor.hpp"

using json = nlohmann::json;

// Florence2Processor class with updated tokenizer and image processor
class Florence2Processor {
public:
    Florence2Processor(const std::string& gguf_path, const std::string& tokenizer_config_path,
                       const std::string& image_config_path, const std::string& mapping_path)
            : ctx_(nullptr), gguf_(nullptr), galloc_(nullptr), graph_(nullptr),
              tokenizer_(tokenizer_config_path), image_processor_(image_config_path, 768) {
        // Validate file existence
        if (!std::filesystem::exists(gguf_path)) throw std::runtime_error("GGUF file not found: " + gguf_path);
        if (!std::filesystem::exists(tokenizer_config_path)) throw std::runtime_error("Tokenizer config file not found: " + tokenizer_config_path);
        if (!std::filesystem::exists(image_config_path)) throw std::runtime_error("Image config file not found: " + image_config_path);
        if (!std::filesystem::exists(mapping_path)) throw std::runtime_error("Tensor mapping file not found: " + mapping_path);

        // Load tensor mapping
        std::ifstream mapping_file(mapping_path);
        json tensor_mapping_json;
        mapping_file >> tensor_mapping_json;
        for (auto& [pytorch_name, gguf_name] : tensor_mapping_json.items()) {
            tensor_mapping_[pytorch_name] = gguf_name;
            tensor_full_names_[gguf_name] = pytorch_name;
        }

        // Initialize GGML context and load GGUF
        initialize_ggml_context(gguf_path);

        // Add special tokens to tokenizer
        std::vector<std::string> special_tokens = {"<od>", "</od>", "<ocr>", "</ocr>", "<cap>", "</cap>"};
        tokenizer_.add_special_tokens(special_tokens);
    }

    ~Florence2Processor() {
        if (galloc_) ggml_gallocr_free(galloc_);
//        if (graph_) ggml_free_graph(graph_);
        if (ctx_) ggml_free(ctx_);
        if (gguf_) gguf_free(gguf_);
    }

    // Process text and image to generate output
    std::string generate(const std::string& text, const cv::Mat& image, int max_length) {
        std::vector<uint32_t> token_ids;
        std::vector<uint32_t> input_ids = tokenizer_.encode(text, true, 256, true, true, "longest");

        // Preprocess image
        ggml_tensor* vision_output = process_image(image);

        // Clear and reset graph
        ggml_graph_clear(graph_);
        reset_graph_allocator();

        // Tokenize text and embed
        ggml_tensor* input_ids_tensor = ggml_new_tensor_2d(ctx_, GGML_TYPE_I64, input_ids.size(), 1);
        int64_t* input_ids_data = static_cast<int64_t*>(ggml_get_data(input_ids_tensor));
        for (size_t i = 0; i < input_ids.size(); ++i) {
            input_ids_data[i] = static_cast<int64_t>(input_ids[i]);
        }

        ggml_tensor* embed_tokens_weight = load_tensor("language_model.model.decoder.embed_tokens.weight");
        ggml_tensor* text_embeddings = ggml_mul_mat(ctx_, embed_tokens_weight, input_ids_tensor);
        ggml_build_forward_expand(graph_, text_embeddings);

        // Concatenate vision and text embeddings
        ggml_tensor* encoder_input = ggml_concat(ctx_, vision_output, text_embeddings, 0);
        ggml_build_forward_expand(graph_, encoder_input);

        // Simplified decoder (for demonstration; expand with full layers)
        ggml_tensor* lm_head_weight = load_tensor("language_model.lm_head.weight");
        ggml_tensor* logits = ggml_mul_mat(ctx_, lm_head_weight, encoder_input);
        ggml_build_forward_expand(graph_, logits);

        // Allocate and compute graph
        if (!ggml_gallocr_alloc_graph(galloc_, graph_)) {
            throw std::runtime_error("Failed to allocate graph memory");
        }
        ggml_graph_compute(graph_, nullptr);

        // Generate tokens
        std::vector<float> logits_vec(vocab_size_);
        ggml_fp16_t* logits_data = static_cast<ggml_fp16_t*>(ggml_get_data(logits));
        for (size_t i = 0; i < vocab_size_; ++i) {
            logits_vec[i] = ggml_fp16_to_fp32(logits_data[i]);
        }

        for (int i = 0; i < max_length; ++i) {
            uint32_t token_id = argmax(logits_vec);
            token_ids.push_back(token_id);
            if (token_id == tokenizer_.get_special_token_id("</s>")) break; // EOS token
            // Simplified: no past key values or iterative decoding here
            break; // For simplicity, stop after one token
        }

        return tokenizer_.decode(token_ids, true);
    }

private:
    struct ggml_context* ctx_;
    struct gguf_context* gguf_;
    struct ggml_gallocr* galloc_;
    struct ggml_cgraph* graph_;
    std::unordered_map<std::string, std::string> tensor_mapping_;
    std::unordered_map<std::string, std::string> tensor_full_names_;
    std::unordered_map<std::string, ggml_tensor*> tensors_;
    BartTokenizerFast tokenizer_;      // Replaced with BartTokenizerFast
    CLIPImageProcessor image_processor_; // Replaced with CLIPImageProcessor
    int vocab_size_ = 51289;
    int t_dec_layers_ = 12;
    int t_enc_layers_ = 12;
    int t_hidden_ = 1024;
    int v_img_size_ = 768;

    void initialize_ggml_context(const std::string& gguf_path) {
        // Calculate required memory size (pseudo-function; replace with actual GGUF API if available)
        size_t required_memory_size = 2ULL * 1024 * 1024 * 1024; // 2GB estimate; adjust based on model size
        struct ggml_init_params params = {
                .mem_size = required_memory_size,
                .mem_buffer = nullptr
        };
        ctx_ = ggml_init(params);
        if (!ctx_) throw std::runtime_error("Failed to initialize GGML context");

        // Load GGUF file
        struct gguf_init_params gguf_params = { .no_alloc = false, .ctx = &ctx_ };
        gguf_ = gguf_init_from_file(gguf_path.c_str(), gguf_params);
        if (!gguf_) throw std::runtime_error("Failed to load GGUF file");

        // Populate tensors
        int n_tensors = gguf_get_n_tensors(gguf_);
        for (int i = 0; i < n_tensors; ++i) {
            const char* name = gguf_get_tensor_name(gguf_, i);
            ggml_tensor* tensor = ggml_get_tensor(ctx_, name);
            if (!tensor) throw std::runtime_error("Failed to get tensor: " + std::string(name));
            tensors_[name] = tensor;
        }

        // Initialize computation graph
        graph_ = ggml_new_graph(ctx_);
        if (!graph_) throw std::runtime_error("Failed to initialize computation graph");

        // Initialize graph allocator
        galloc_ = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        if (!galloc_) throw std::runtime_error("Failed to initialize graph allocator");
    }

    void reset_graph_allocator() {
        if (galloc_) ggml_gallocr_free(galloc_);
        galloc_ = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        if (!galloc_ || !ggml_gallocr_reserve(galloc_, graph_)) {
            throw std::runtime_error("Failed to reset graph allocator");
        }
    }

    ggml_tensor* load_tensor(const std::string& full_name) {
        auto it = tensor_mapping_.find(full_name);
        if (it == tensor_mapping_.end()) throw std::runtime_error("Tensor not found in mapping: " + full_name);
        std::string gguf_name = it->second;
        auto tensor_it = tensors_.find(gguf_name);
        if (tensor_it == tensors_.end()) throw std::runtime_error("Tensor not found in GGUF: " + gguf_name);
        return tensor_it->second;
    }

    ggml_tensor* process_image(const cv::Mat& image) {
        // Preprocess image using CLIPImageProcessor
        torch::Tensor image_tensor = image_processor_.Preprocess(image); // Returns [1, 3, 768, 768]
        assert(image_tensor.dim() == 4 && image_tensor.size(0) == 1 && image_tensor.size(1) == 3 &&
               image_tensor.size(2) == 768 && image_tensor.size(3) == 768 && "Unexpected image tensor shape");

        // Convert torch::Tensor to GGML tensor
        image_tensor = image_tensor.squeeze(0).permute({2, 1, 0}); // [768, 768, 3]
        ggml_tensor* pixel_values = ggml_new_tensor_4d(ctx_, GGML_TYPE_F16, 768, 768, 3, 1);
        ggml_fp16_t* pixel_data = static_cast<ggml_fp16_t*>(ggml_get_data(pixel_values));
        float* tensor_data = image_tensor.data_ptr<float>();
        size_t expected_size = 768 * 768 * 3;
        for (size_t i = 0; i < expected_size; ++i) {
            pixel_data[i] = ggml_fp32_to_fp16(tensor_data[i]);
        }

        // Simplified vision tower (convolution stages)
        ggml_tensor* conv0_weight = load_tensor("vision_tower.convs.0.proj.weight");
        ggml_tensor* conv0_bias = load_tensor("vision_tower.convs.0.proj.bias");
        ggml_tensor* conv0 = ggml_conv_2d(ctx_, conv0_weight, pixel_values, 4, 4, 3, 3, 1, 1);
        ggml_tensor* conv0_with_bias = ggml_add(ctx_, conv0, conv0_bias);
        ggml_tensor* norm0 = ggml_group_norm(ctx_, conv0_with_bias, 256, 1e-5f);
        ggml_build_forward_expand(graph_, norm0);

        // Simplified projection (expand with full vision tower as needed)
        ggml_tensor* vision_proj_weight = load_tensor("image_projection");
        ggml_tensor* pooled = ggml_mean(ctx_, norm0); // Simplified pooling
        ggml_tensor* vision_output = ggml_mul_mat(ctx_, vision_proj_weight, pooled);
        ggml_build_forward_expand(graph_, vision_output);

        return vision_output;
    }

    uint32_t argmax(const std::vector<float>& vec) {
        float max_val = vec[0];
        uint32_t max_idx = 0;
        for (size_t i = 1; i < vec.size(); ++i) {
            if (vec[i] > max_val) {
                max_val = vec[i];
                max_idx = static_cast<uint32_t>(i);
            }
        }
        return max_idx;
    }
};

// Main function for testing
int main() {
    try {
        Florence2Processor processor(
                "florence2.gguf",
                "tokenizer.json",
                "preprocessor_config.json",
                "tensors_mapping.json"
        );

        cv::Mat image = cv::imread("puma.png");
        if (image.empty()) throw std::runtime_error("Failed to load test image");

        std::string text = "<CAPTION>";
        std::string output = processor.generate(text, image, 50);
        std::cout << "Generated output: " << output << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
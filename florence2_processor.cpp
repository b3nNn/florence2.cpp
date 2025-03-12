#include "florence2_processor.hpp"
#include <ggml.h>
#include <ggml-cpu.h>
#include <gguf.h>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp> // Ensure OpenCV header is included

namespace Florence2Processor {

    static const std::unordered_map<std::string, std::string> task_prompts = {
            {"<CAPTION>", "<cap>"},
            {"<OD>", "<od>"},
            {"<OCR>", "<ocr>"}
    };

    static const int image_seq_length = 833;

    Florence2Processor::Florence2Processor(const std::string& gguf_path, const std::string& tokenizer_json_path, const std::string& image_config_path)
            : tokenizer(tokenizer_json_path), image_processor(image_config_path, 768) {
        std::cout << "Checking files...\n";
        if (!std::filesystem::exists(gguf_path)) throw std::runtime_error("GGUF file not found: " + gguf_path);
        if (!std::filesystem::exists(tokenizer_json_path)) throw std::runtime_error("Tokenizer file not found: " + tokenizer_json_path);
        if (!std::filesystem::exists(image_config_path)) throw std::runtime_error("Image config file not found: " + image_config_path);

        cv::setNumThreads(0); // Disable OpenCV parallel backends

        // Expected values from config.json and convert.py
        const uint32_t expected_vocab_size = 51289;
        const uint32_t expected_t_dec_layers = 12;
        const uint32_t expected_t_enc_layers = 12;
        const uint32_t expected_t_heads = 16;
        const uint32_t expected_t_hidden = 1024;
        const uint32_t expected_t_ffn_dim = 4096;
        const uint32_t expected_v_img_size = 768;

        // Load and validate metadata
        gguf_init_params gguf_params = { .no_alloc = true, .ctx = NULL };
        gguf_context* temp_ctx = gguf_init_from_file(gguf_path.c_str(), gguf_params);
        if (!temp_ctx) throw std::runtime_error("Failed to initialize GGUF context from file: " + gguf_path);

        vocab_size = get_metadata_int(temp_ctx, "vocab_size");
        std::cout << "Loaded vocab_size: " << vocab_size << "\n";
        if (vocab_size != expected_vocab_size) {
            throw std::runtime_error("Metadata mismatch: vocab_size = " + std::to_string(vocab_size) +
                                     ", expected " + std::to_string(expected_vocab_size));
        }

        t_dec_layers = get_metadata_int(temp_ctx, "t_dec_layers");
        std::cout << "Loaded t_dec_layers: " << t_dec_layers << "\n";
        if (t_dec_layers <= 0 || static_cast<uint32_t>(t_dec_layers) != expected_t_dec_layers) {
            throw std::runtime_error("Metadata mismatch: t_dec_layers = " + std::to_string(t_dec_layers) +
                                     ", expected " + std::to_string(expected_t_dec_layers));
        }

        t_enc_layers = get_metadata_int(temp_ctx, "t_enc_layers");
        std::cout << "Loaded t_enc_layers: " << t_enc_layers << "\n";
        if (t_enc_layers <= 0 || static_cast<uint32_t>(t_enc_layers) != expected_t_enc_layers) {
            throw std::runtime_error("Metadata mismatch: t_enc_layers = " + std::to_string(t_enc_layers) +
                                     ", expected " + std::to_string(expected_t_enc_layers));
        }

        t_heads = get_metadata_int(temp_ctx, "t_heads");
        std::cout << "Loaded t_heads: " << t_heads << "\n";
        if (t_heads <= 0 || static_cast<uint32_t>(t_heads) != expected_t_heads) {
            throw std::runtime_error("Metadata mismatch: t_heads = " + std::to_string(t_heads) +
                                     ", expected " + std::to_string(expected_t_heads));
        }

        t_hidden = get_metadata_int(temp_ctx, "t_hidden");
        std::cout << "Loaded t_hidden: " << t_hidden << "\n";
        if (t_hidden <= 0 || static_cast<uint32_t>(t_hidden) != expected_t_hidden) {
            throw std::runtime_error("Metadata mismatch: t_hidden = " + std::to_string(t_hidden) +
                                     ", expected " + std::to_string(expected_t_hidden));
        }

        t_ffn_dim = get_metadata_int(temp_ctx, "t_ffn_dim");
        std::cout << "Loaded t_ffn_dim: " << t_ffn_dim << "\n";
        if (t_ffn_dim <= 0 || static_cast<uint32_t>(t_ffn_dim) != expected_t_ffn_dim) {
            throw std::runtime_error("Metadata mismatch: t_ffn_dim = " + std::to_string(t_ffn_dim) +
                                     ", expected " + std::to_string(expected_t_ffn_dim));
        }

        v_img_size = get_metadata_int(temp_ctx, "v_img_size");
        std::cout << "Loaded v_img_size: " << v_img_size << "\n";
        if (v_img_size <= 0 || static_cast<uint32_t>(v_img_size) != expected_v_img_size) {
            throw std::runtime_error("Metadata mismatch: v_img_size = " + std::to_string(v_img_size) +
                                     ", expected " + std::to_string(expected_v_img_size));
        }
        gguf_free(temp_ctx);

        // Initialize gguf ctx with pre-allocated memory
        initialize_ggml_context(gguf_path);

        // Add special tokens
        std::vector<std::string> special_tokens;
        for (int i = 0; i < 1000; i++) special_tokens.push_back("<loc_" + std::to_string(i) + ">");
        special_tokens.insert(special_tokens.end(), {"<od>", "</od>", "<ocr>", "</ocr>", "<cap>", "</cap>", "<ncap>", "</ncap>",
                                                     "<dcap>", "</dcap>", "<grounding>", "</grounding>", "<seg>", "</seg>",
                                                     "<sep>", "<region_cap>", "</region_cap>", "<region_to_desciption>",
                                                     "</region_to_desciption>", "<proposal>", "</proposal>", "<poly>", "</poly>", "<and>"});
        tokenizer.add_special_tokens(special_tokens);
        std::cout << "Processor initialized\n";
    }

    Florence2Processor::~Florence2Processor() {
        gguf_free(gguf_ctx);
        ggml_free(ctx);
    }

    void Florence2Processor::initialize_ggml_context(const std::string& gguf_path) {
        // Step 1: Calculate GGUF tensor memory (metadata + rough tensor estimate)
        gguf_init_params gguf_params = { .no_alloc = true, .ctx = NULL };
        gguf_context* temp_ctx = gguf_init_from_file(gguf_path.c_str(), gguf_params);
        if (!temp_ctx) throw std::runtime_error("Failed to initialize GGUF context from file: " + gguf_path);

        size_t gguf_size = gguf_get_meta_size(temp_ctx);
        int n_tensors = gguf_get_n_tensors(temp_ctx);
        // Rough estimate: assume average tensor size based on prior runs (~1.83 GB total for tensors)
        size_t avg_tensor_size = (1831148768ULL / n_tensors); // From previous "used" value
        gguf_size += n_tensors * (avg_tensor_size + GGML_TENSOR_SIZE);

        // Step 2: Estimate runtime tensor memory from process()
        size_t runtime_size = 0;
        runtime_size += 768 * 768 * 3 * 1 * 2 + GGML_TENSOR_SIZE; // pixel_values
        runtime_size += 256 * 1 * 8 + GGML_TENSOR_SIZE; // input_ids_tensor
        runtime_size += 5 * 1 * 8 + GGML_TENSOR_SIZE; // decoder_input_ids_tensor (max_length=5)
        size_t past_tensor_size = 64 * 833 * 16 * 1 * 2; // ~1.7 MB per tensor
        runtime_size += t_dec_layers * 4 * (past_tensor_size + GGML_TENSOR_SIZE); // past key/values

        // Step 3: Add graph overhead
        size_t graph_size = 455 * GGML_TENSOR_SIZE; // ~116 KB

        // Step 4: Total size with safety margin
        size_t total_size = gguf_size + runtime_size + graph_size;
        total_size += 64ULL * 1024 * 1024; // 64 MB safety buffer
        total_size = GGML_PAD(total_size, GGML_MEM_ALIGN);

        gguf_free(temp_ctx); // Free temporary context

        // Step 5: Initialize GGML context
        ggml_init_params params = { .mem_size = total_size, .mem_buffer = NULL };
        ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("Failed to initialize GGML context with " + std::to_string(total_size) + " bytes");
        std::cout << "Allocated GGML memory: " << total_size << " bytes\n";

        // Step 6: Load GGUF with tensor allocation into GGML context
        gguf_params = { .no_alloc = false, .ctx = &ctx };
        gguf_ctx = gguf_init_from_file(gguf_path.c_str(), gguf_params);
        if (!gguf_ctx) throw std::runtime_error("Failed to initialize GGUF context from file: " + gguf_path);

        // Step 7: Populate tensor map with all exported tensors
        for (int i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(gguf_ctx, i);
            ggml_tensor* tensor = ggml_get_tensor(ctx, name);
            if (!tensor) {
                throw std::runtime_error("Failed to retrieve tensor from GGML context: " + std::string(name));
            }
            tensors[name] = tensor;
            std::cout << "Loaded tensor: " << name << "\n"; // Debug output
        }

        size_t used_mem = ggml_used_mem(ctx);
        std::cout << "GGML memory used after loading: " << used_mem << " bytes\n";
    }

    int Florence2Processor::get_metadata_int(gguf_context* ctx, const std::string& key) {
        int idx = gguf_find_key(ctx, key.c_str());
        if (idx == -1) throw std::runtime_error("Metadata key not found: " + key);
        if (gguf_get_kv_type(ctx, idx) != GGUF_TYPE_UINT32) throw std::runtime_error("Metadata key " + key + " is not uint32");
        return gguf_get_val_u32(ctx, idx);
    }

    ggml_tensor* Florence2Processor::load_tensor(const std::string& name) {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("Tensor not found: " + name);
        return it->second;
    }

    ModelOutput Florence2Processor::process(const std::string& text, const cv::Mat& image, const std::vector<int64_t>& decoder_input_ids,
                                            const std::vector<PastKeyValueLayer>& past_key_values) {
        std::cout << "Processing text: " << text << "\n";
        std::string prompt = task_prompts.count(text) ? task_prompts.at(text) : text;
        std::vector<uint32_t> token_ids = tokenizer.encode(prompt, true, 256 - image_seq_length, true, true, "longest");
        std::vector<int64_t> input_ids(token_ids.begin(), token_ids.end());
        input_ids.resize(256, 0);

        if (image.empty()) throw std::runtime_error("Image is empty—check file path or cv::imread");
        torch::Tensor image_tensor = image_processor.Preprocess(image);
        std::vector<float> image_data(image_tensor.data_ptr<float>(), image_tensor.data_ptr<float>() + image_tensor.numel());

        ggml_tensor* pixel_values = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 768, 768, 3, 1);
        float* pixel_data = static_cast<float*>(ggml_get_data(pixel_values));
        std::copy(image_data.begin(), image_data.end(), pixel_data);

        ggml_tensor* input_ids_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 256, 1);
        int64_t* input_ids_data = static_cast<int64_t*>(ggml_get_data(input_ids_tensor));
        std::copy(input_ids.begin(), input_ids.end(), input_ids_data);

        ggml_tensor* decoder_input_ids_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, decoder_input_ids.size(), 1);
        int64_t* decoder_ids_data = static_cast<int64_t*>(ggml_get_data(decoder_input_ids_tensor));
        std::copy(decoder_input_ids.begin(), decoder_input_ids.end(), decoder_ids_data);

        std::vector<ggml_tensor*> past_self_keys(t_dec_layers), past_self_values(t_dec_layers);
        std::vector<ggml_tensor*> past_cross_keys(t_dec_layers), past_cross_values(t_dec_layers);
        size_t past_seq_len = past_key_values.empty() ? 0 : past_key_values[0].self_key.size() / (16 * 64);
        if (!past_key_values.empty()) {
            for (int i = 0; i < t_dec_layers; ++i) {
                past_self_keys[i] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, past_seq_len, 16, 1);
                past_self_values[i] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, past_seq_len, 16, 1);
                past_cross_keys[i] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 833, 16, 1);
                past_cross_values[i] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 833, 16, 1);

                float* self_key_data = static_cast<float*>(ggml_get_data(past_self_keys[i]));
                float* self_value_data = static_cast<float*>(ggml_get_data(past_self_values[i]));
                float* cross_key_data = static_cast<float*>(ggml_get_data(past_cross_keys[i]));
                float* cross_value_data = static_cast<float*>(ggml_get_data(past_cross_values[i]));
                std::copy(past_key_values[i].self_key.begin(), past_key_values[i].self_key.end(), self_key_data);
                std::copy(past_key_values[i].self_value.begin(), past_key_values[i].self_value.end(), self_value_data);
                std::copy(past_key_values[i].cross_key.begin(), past_key_values[i].cross_key.end(), cross_key_data);
                std::copy(past_key_values[i].cross_value.begin(), past_key_values[i].cross_value.end(), cross_value_data);
            }
        }

        ggml_tensor* v_pjw = load_tensor("v_pjw");
        ggml_tensor* vision_output = ggml_mul_mat(ctx, v_pjw, pixel_values);

        ggml_tensor* encoder_output = input_ids_tensor;
        for (int i = 0; i < t_enc_layers; ++i) {
            ggml_tensor* te_qw = load_tensor("te" + std::to_string(i) + "a_qw");
            encoder_output = ggml_mul_mat(ctx, te_qw, encoder_output);
        }

        std::vector<ggml_tensor*> new_self_keys(t_dec_layers), new_self_values(t_dec_layers);
        std::vector<ggml_tensor*> new_cross_keys(t_dec_layers), new_cross_values(t_dec_layers);
        ggml_tensor* decoder_output = decoder_input_ids_tensor;
        for (int i = 0; i < t_dec_layers; ++i) {
            ggml_tensor* t_qw = load_tensor("t" + std::to_string(i) + "a_qw");
            ggml_tensor* t_c_qw = load_tensor("t" + std::to_string(i) + "c_qw");
            decoder_output = ggml_mul_mat(ctx, t_qw, decoder_output);
            decoder_output = ggml_mul_mat(ctx, t_c_qw, encoder_output);
            new_self_keys[i] = decoder_output;
            new_self_values[i] = decoder_output;
            new_cross_keys[i] = encoder_output;
            new_cross_values[i] = encoder_output;
        }

        ggml_tensor* t_out_w = load_tensor("t_out_w");
        ggml_tensor* logits = ggml_mul_mat(ctx, t_out_w, decoder_output);

        ModelOutput output;
        output.logits.resize(vocab_size);
        float* logits_data = static_cast<float*>(ggml_get_data(logits));
        std::copy(logits_data, logits_data + vocab_size, output.logits.begin());
        output.past_key_values.resize(t_dec_layers);
        for (int i = 0; i < t_dec_layers; ++i) {
            size_t self_size = new_self_keys[i]->ne[0] * new_self_keys[i]->ne[1] * new_self_keys[i]->ne[2];
            size_t cross_size = new_cross_keys[i]->ne[0] * new_cross_keys[i]->ne[1] * new_cross_keys[i]->ne[2];
            output.past_key_values[i].self_key.resize(self_size);
            output.past_key_values[i].self_value.resize(self_size);
            output.past_key_values[i].cross_key.resize(cross_size);
            output.past_key_values[i].cross_value.resize(cross_size);
            float* self_key_data = static_cast<float*>(ggml_get_data(new_self_keys[i]));
            float* self_value_data = static_cast<float*>(ggml_get_data(new_self_values[i]));
            float* cross_key_data = static_cast<float*>(ggml_get_data(new_cross_keys[i]));
            float* cross_value_data = static_cast<float*>(ggml_get_data(new_cross_values[i]));
            std::copy(self_key_data, self_key_data + self_size, output.past_key_values[i].self_key.begin());
            std::copy(self_value_data, self_value_data + self_size, output.past_key_values[i].self_value.begin());
            std::copy(cross_key_data, cross_key_data + cross_size, output.past_key_values[i].cross_key.begin());
            std::copy(cross_value_data, cross_value_data + cross_size, output.past_key_values[i].cross_value.begin());
        }

        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, logits);
        ggml_graph_compute_with_ctx(ctx, gf, GGML_DEFAULT_N_THREADS);
        return output;
    }

    std::string Florence2Processor::decode(const std::vector<uint32_t>& token_ids, bool skip_special_tokens) {
        return tokenizer.decode(token_ids, skip_special_tokens);
    }

    std::string Florence2Processor::generate(const std::string& text, const cv::Mat& image, int max_length) {
        std::vector<uint32_t> token_ids;
        std::vector<int64_t> decoder_input_ids = {0};
        std::vector<PastKeyValueLayer> past_key_values;

        for (int i = 0; i < max_length; ++i) {
            ModelOutput output = process(text, image, decoder_input_ids, past_key_values);
            float max_val = output.logits[0];
            uint32_t token_id = 0;
            for (size_t j = 1; j < output.logits.size(); ++j) {
                if (output.logits[j] > max_val) {
                    max_val = output.logits[j];
                    token_id = static_cast<uint32_t>(j);
                }
            }
            token_ids.push_back(token_id);
            decoder_input_ids.push_back(token_id);
            past_key_values = std::move(output.past_key_values);
            std::cout << "Step " << i << " - Token ID: " << token_id << "\n";
            if (token_id == 2) break;
        }
        return decode(token_ids, true);
    }

} // namespace Florence2Processor

int main() {
    try {
        std::string gguf_path = "florence2.gguf";
        std::string tokenizer_path = "tokenizer.json";
        std::string image_config_path = "preprocessor_config.json";
        if (!std::filesystem::exists("puma.png")) {
            std::cerr << "puma.png not found in current directory\n";
            return 1;
        }
        Florence2Processor::Florence2Processor processor(gguf_path, tokenizer_path, image_config_path);
        cv::Mat image = cv::imread("puma.png");
        if (image.empty()) {
            std::cerr << "Failed to load puma.png—check file integrity\n";
            return 1;
        }
        std::string caption = processor.generate("<CAPTION>", image, 5);
        std::cout << "Caption: " << caption << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
#include "florence2_processor.hpp"
#include <ggml.h>
#include <ggml-cpu.h>
#include <gguf.h>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace Florence2Processor {

    static const std::unordered_map<std::string, std::string> task_prompts = {
            {"<CAPTION>", "<cap>"}, {"<OD>", "<od>"}, {"<OCR>", "<ocr>"}
    };

    static const int image_seq_length = 833;

    Florence2Processor::Florence2Processor(const std::string& gguf_path, const std::string& tokenizer_json_path, const std::string& image_config_path)
            : tokenizer(tokenizer_json_path), image_processor(image_config_path, 768) {
        std::cout << "Checking files...\n";
        if (!std::filesystem::exists(gguf_path)) throw std::runtime_error("GGUF file not found: " + gguf_path);
        if (!std::filesystem::exists(tokenizer_json_path)) throw std::runtime_error("Tokenizer file not found: " + tokenizer_json_path);
        if (!std::filesystem::exists(image_config_path)) throw std::runtime_error("Image config file not found: " + image_config_path);

        cv::setNumThreads(0);

        const uint32_t expected_vocab_size = 51289;
        const uint32_t expected_t_dec_layers = 12;
        const uint32_t expected_t_enc_layers = 12;
        const uint32_t expected_t_heads = 16;
        const uint32_t expected_t_hidden = 1024;
        const uint32_t expected_t_ffn_dim = 4096;
        const uint32_t expected_v_img_size = 768;

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

        initialize_ggml_context(gguf_path);

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

    struct TensorMetadata {
        std::string name;
        int ndims;
        int64_t dims[4];
        ggml_type type;
    };

    void Florence2Processor::initialize_ggml_context(const std::string& gguf_path) {
        gguf_init_params temp_params = { .no_alloc = true, .ctx = NULL };
        gguf_context* temp_gguf_ctx = gguf_init_from_file(gguf_path.c_str(), temp_params);
        if (!temp_gguf_ctx) throw std::runtime_error("Failed to initialize temporary GGUF context from file: " + gguf_path);

        size_t gguf_size = gguf_get_meta_size(temp_gguf_ctx);
        int n_tensors = gguf_get_n_tensors(temp_gguf_ctx);
        size_t gguf_tensor_size = 1831148768ULL;

        size_t runtime_size = 0;
        runtime_size += 768 * 768 * 3 * 1 * 2 + GGML_TENSOR_SIZE;
        runtime_size += 192 * 192 * 256 * 1 * 2 + GGML_TENSOR_SIZE;
        runtime_size += 192 * 192 * 256 * 1 * 2 + GGML_TENSOR_SIZE;
        runtime_size += 192 * 192 * 256 * 1 * 2 + GGML_TENSOR_SIZE;
        runtime_size += 96 * 96 * 512 * 1 * 2 + GGML_TENSOR_SIZE;
        runtime_size += 96 * 96 * 512 * 1 * 2 + GGML_TENSOR_SIZE;
        runtime_size += 96 * 96 * 512 * 1 * 2 + GGML_TENSOR_SIZE;
        runtime_size += 48 * 48 * 1024 * 1 * 2 + GGML_TENSOR_SIZE;
        runtime_size += 48 * 48 * 1024 * 1 * 2 + GGML_TENSOR_SIZE;
        runtime_size += 48 * 48 * 1024 * 1 * 2 + GGML_TENSOR_SIZE;
        runtime_size += 256 * 1 * 8 + GGML_TENSOR_SIZE;
        runtime_size += 5 * 1 * 8 + GGML_TENSOR_SIZE;

        size_t graph_size = 300ULL * 1024 * 1024;
        size_t total_size = gguf_tensor_size + runtime_size + graph_size + gguf_size;
        total_size += 512ULL * 1024 * 1024;
        total_size = GGML_PAD(total_size, GGML_MEM_ALIGN);

        std::cout << "Calculated GGUF size: " << gguf_tensor_size << " bytes (~" << gguf_tensor_size / (1024.0 * 1024 * 1024) << " GB)\n";
        std::cout << "Calculated runtime size: " << runtime_size << " bytes (~" << runtime_size / (1024.0 * 1024) << " MB)\n";
        std::cout << "Calculated graph size: " << graph_size << " bytes (~" << graph_size / (1024.0 * 1024) << " MB)\n";
        std::cout << "Total allocated size: " << total_size << " bytes (~" << total_size / (1024.0 * 1024 * 1024) << " GB)\n";

        ggml_init_params params = { .mem_size = total_size, .mem_buffer = NULL };
        ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("Failed to initialize GGML context with " + std::to_string(total_size) + " bytes");

        size_t used_mem = ggml_used_mem(ctx);
        std::cout << "Memory used after ggml_init: " << used_mem << " bytes\n";

        pixel_values_prealloc = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 768, 768, 3, 1);
        conv0_output = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 192, 192, 256, 1);
        conv0_bias = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 192, 192, 256, 1);
        norm0_prealloc = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 192, 192, 256, 1);
        conv1_output = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 96, 96, 512, 1);
        conv1_bias = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 96, 96, 512, 1);
        norm1_prealloc = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 96, 96, 512, 1);
        conv2_output = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 48, 48, 1024, 1);
        conv2_bias = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 48, 48, 1024, 1);
        norm2_prealloc = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 48, 48, 1024, 1);
        input_ids_tensor_prealloc = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 256, 1);
        decoder_input_ids_tensor_prealloc = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 5, 1);

        used_mem = ggml_used_mem(ctx);
        std::cout << "Memory used after runtime tensor allocation: " << used_mem << " bytes (~" << used_mem / (1024.0 * 1024) << " MB)\n";

        // Move graph allocation earlier and reserve memory
        galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        if (!galloc) throw std::runtime_error("Failed to initialize GGML graph allocator");
        std::cout << "Initialized GGML graph allocator\n";

        graph = ggml_new_graph(ctx);
        if (!graph) throw std::runtime_error("Failed to initialize computation graph");

        // Reserve graph memory to pre-allocate space
        ggml_gallocr_reserve(galloc, graph);

        std::vector<std::pair<std::string, ggml_type>> gguf_tensors;
        gguf_tensors.reserve(n_tensors);
        for (int i = 0; i < n_tensors; i++) {
            std::string name = gguf_get_tensor_name(temp_gguf_ctx, i);
            ggml_type type = static_cast<ggml_type>(gguf_get_tensor_type(temp_gguf_ctx, i));
            gguf_tensors.emplace_back(name, type);
        }

        for (const auto& [name, type] : gguf_tensors) {
            ggml_tensor* tensor = ggml_new_tensor_1d(ctx, type, 1);
            if (!tensor) throw std::runtime_error("Failed to pre-allocate tensor: " + name);
            tensors[name] = tensor;
        }

        used_mem = ggml_used_mem(ctx);
        std::cout << "Memory used after GGUF tensor pre-allocation: " << used_mem << " bytes (~" << used_mem / (1024.0 * 1024 * 1024) << " GB)\n";

        gguf_init_params gguf_params = { .no_alloc = false, .ctx = &ctx };
        gguf_ctx = gguf_init_from_file(gguf_path.c_str(), gguf_params);
        if (!gguf_ctx) throw std::runtime_error("Failed to initialize GGUF context from file: " + gguf_path);

        used_mem = ggml_used_mem(ctx);
        std::cout << "Memory used after GGUF loading: " << used_mem << " bytes (~" << used_mem / (1024.0 * 1024 * 1024) << " GB)\n";

        if (used_mem > total_size) {
            throw std::runtime_error("Memory usage (" + std::to_string(used_mem) + ") exceeds allocated size (" + std::to_string(total_size) + ")");
        }
        std::cout << "Remaining memory: " << (total_size - used_mem) << " bytes (~" << (total_size - used_mem) / (1024.0 * 1024) << " MB)\n";

        gguf_free(temp_gguf_ctx);
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

        float* pixel_data = static_cast<float*>(ggml_get_data(pixel_values_prealloc));
        std::copy(image_data.begin(), image_data.end(), pixel_data);

        // Reset graph for reuse
        ggml_graph_clear(graph);

        // Stage 0
        ggml_tensor* v_c0w = load_tensor("v_c0w"); // [256, 3, 7, 7]
        ggml_tensor* v_c0b = load_tensor("v_c0b"); // [256]
        ggml_tensor* conv0 = ggml_conv_2d(ctx, v_c0w, pixel_values_prealloc, 4, 4, 3, 3, 1, 1);
        std::cout << "Allocated conv0\n";
        ggml_cpy(ctx, conv0, conv0_output);
        ggml_add_inplace(ctx, conv0_output, v_c0b);
        ggml_cpy(ctx, ggml_group_norm(ctx, conv0_output, 256, 1e-5f), norm0_prealloc);
        ggml_build_forward_expand(graph, norm0_prealloc);

        // Stage 1
        ggml_tensor* v_c1w = load_tensor("v_c1w"); // [512, 256, 3, 3]
        ggml_tensor* v_c1b = load_tensor("v_c1b"); // [512]
        ggml_tensor* conv1 = ggml_conv_2d(ctx, v_c1w, norm0_prealloc, 2, 2, 1, 1, 1, 1);
        std::cout << "Allocated conv1\n";
        ggml_cpy(ctx, conv1, conv1_output);
        ggml_add_inplace(ctx, conv1_output, v_c1b);
        ggml_cpy(ctx, ggml_group_norm(ctx, conv1_output, 512, 1e-5f), norm1_prealloc);
        ggml_build_forward_expand(graph, norm1_prealloc);

        // Stage 2
        ggml_tensor* v_c2w = load_tensor("v_c2w"); // [1024, 512, 3, 3]
        ggml_tensor* v_c2b = load_tensor("v_c2b"); // [1024]
        ggml_tensor* conv2 = ggml_conv_2d(ctx, v_c2w, norm1_prealloc, 2, 2, 1, 1, 1, 1);
        std::cout << "Allocated conv2\n";
        ggml_cpy(ctx, conv2, conv2_output);
        ggml_add_inplace(ctx, conv2_output, v_c2b);
        ggml_cpy(ctx, ggml_group_norm(ctx, conv2_output, 1024, 1e-5f), norm2_prealloc);
        ggml_build_forward_expand(graph, norm2_prealloc);

        // Placeholder for Stage 3 and blocks
        ggml_tensor* v_pjw = load_tensor("v_pjw");
        ggml_tensor* vision_output = ggml_mul_mat(ctx, v_pjw, norm2_prealloc);
        ggml_build_forward_expand(graph, vision_output);

        int64_t* input_ids_data = static_cast<int64_t*>(ggml_get_data(input_ids_tensor_prealloc));
        std::copy(input_ids.begin(), input_ids.end(), input_ids_data);

        ggml_tensor* encoder_output = input_ids_tensor_prealloc;
        ggml_tensor* t_out_w = load_tensor("t_out_w");
        ggml_tensor* logits = ggml_mul_mat(ctx, t_out_w, encoder_output);
        ggml_build_forward_expand(graph, logits);

        // Allocate graph memory (already reserved)
        if (!ggml_gallocr_alloc_graph(galloc, graph)) {
            throw std::runtime_error("Failed to allocate graph memory");
        }

        // Compute the graph
        ggml_graph_compute_with_ctx(ctx, graph, GGML_DEFAULT_N_THREADS);

        ModelOutput output;
        output.logits.resize(vocab_size);
        float* logits_data = static_cast<float*>(ggml_get_data(logits));
        std::copy(logits_data, logits_data + vocab_size, output.logits.begin());
        output.past_key_values.resize(t_dec_layers);

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
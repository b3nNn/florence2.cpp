#include "florence2_processor.hpp"
#include <filesystem>
#include <iostream>

namespace Florence2Processor {

    static const std::unordered_map<std::string, std::string> task_prompts = {
            {"<CAPTION>", "<cap>"}, {"<OD>", "<od>"}, {"<OCR>", "<ocr>"}
    };

    static const int image_seq_length = 833;

    Florence2Processor::Florence2Processor(const std::string& gguf_path, const std::string& tokenizer_json_path, const std::string& image_config_path)
            : ctx(nullptr), gguf_ctx(nullptr), galloc(nullptr), graph(nullptr),
              pixel_values_prealloc(nullptr), conv0_output(nullptr), conv0_bias(nullptr), norm0_prealloc(nullptr),
              conv1_output(nullptr), conv1_bias(nullptr), norm1_prealloc(nullptr),
              conv2_output(nullptr), conv2_bias(nullptr), norm2_prealloc(nullptr),
              input_ids_tensor_prealloc(nullptr), decoder_input_ids_tensor_prealloc(nullptr),
              tokenizer(tokenizer_json_path), image_processor(image_config_path, 768) {
        std::cout << "Checking files...\n";
        if (!std::filesystem::exists(gguf_path)) throw std::runtime_error("GGUF file not found: " + gguf_path);
        if (!std::filesystem::exists(tokenizer_json_path)) throw std::runtime_error("Tokenizer file not found: " + tokenizer_json_path);
        if (!std::filesystem::exists(image_config_path)) throw std::runtime_error("Image config file not found: " + image_config_path);

        cv::setNumThreads(0);

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
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
        if (gguf_ctx) ggml_free(gguf_ctx);
    }

    void Florence2Processor::initialize_ggml_context(const std::string& gguf_path) {
        // GGUF context
        size_t gguf_tensor_size = 1831148768ULL;
        ggml_init_params gguf_params = { .mem_size = gguf_tensor_size + 1024 * 1024, .mem_buffer = NULL };
        gguf_ctx = ggml_init(gguf_params);
        if (!gguf_ctx) throw std::runtime_error("Failed to initialize GGUF context");

        gguf_init_params temp_params = { .no_alloc = true, .ctx = NULL };
        gguf_context* temp_gguf_ctx = gguf_init_from_file(gguf_path.c_str(), temp_params);
        if (!temp_gguf_ctx) throw std::runtime_error("Failed to initialize temporary GGUF context");

        size_t gguf_size = gguf_get_meta_size(temp_gguf_ctx);
        int n_tensors = gguf_get_n_tensors(temp_gguf_ctx);

        gguf_init_params load_params = { .no_alloc = false, .ctx = &gguf_ctx };
        gguf_context* file_ctx = gguf_init_from_file(gguf_path.c_str(), load_params);
        if (!file_ctx) throw std::runtime_error("Failed to load GGUF file into gguf_ctx");

        vocab_size = get_metadata_int(temp_gguf_ctx, "vocab_size");
        t_dec_layers = get_metadata_int(temp_gguf_ctx, "t_dec_layers");
        t_enc_layers = get_metadata_int(temp_gguf_ctx, "t_enc_layers");
        t_heads = get_metadata_int(temp_gguf_ctx, "t_heads");
        t_hidden = get_metadata_int(temp_gguf_ctx, "t_hidden");
        t_ffn_dim = get_metadata_int(temp_gguf_ctx, "t_ffn_dim");
        v_img_size = get_metadata_int(temp_gguf_ctx, "v_img_size");

        for (int i = 0; i < n_tensors; i++) {
            std::string name = gguf_get_tensor_name(temp_gguf_ctx, i);
            ggml_tensor* tensor = ggml_get_tensor(gguf_ctx, name.c_str());
            if (!tensor) throw std::runtime_error("Failed to get tensor: " + name);
            tensors[name] = tensor;
        }
        gguf_free(temp_gguf_ctx);
        gguf_free(file_ctx);

        // Main context
        size_t runtime_size = 0;
        runtime_size += 768 * 768 * 3 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 192 * 192 * 256 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 192 * 192 * 256 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 192 * 192 * 256 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 96 * 96 * 512 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 96 * 96 * 512 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 96 * 96 * 512 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 48 * 48 * 1024 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 48 * 48 * 1024 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 48 * 48 * 1024 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 256 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 5 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 192 * 192 * 256 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 96 * 96 * 512 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 48 * 48 * 1024 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 7 * 7 * 3 * 256 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 256 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 3 * 3 * 256 * 512 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 512 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 3 * 3 * 512 * 1024 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 1024 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;

        size_t graph_size = 1800ULL * 1024 * 1024;
        size_t total_size = runtime_size + graph_size + 1024ULL * 1024 * 1024;
        total_size = GGML_PAD(total_size, GGML_MEM_ALIGN);

        ggml_init_params params = { .mem_size = total_size, .mem_buffer = NULL };
        ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("Failed to initialize main GGML context");

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
        input_ids_tensor_prealloc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 1);
        decoder_input_ids_tensor_prealloc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 5, 1);

        graph = ggml_new_graph(ctx);
        if (!graph) throw std::runtime_error("Failed to initialize computation graph");

        reset_graph_allocator();

        std::cout << "GGUF ctx size: " << gguf_tensor_size << " bytes (~" << gguf_tensor_size / (1024.0 * 1024 * 1024) << " GB)\n";
        std::cout << "Main ctx size: " << total_size << " bytes (~" << total_size / (1024.0 * 1024 * 1024) << " GB)\n";
    }

    void Florence2Processor::reset_graph_allocator() {
        if (galloc) {
            ggml_gallocr_free(galloc);
            galloc = nullptr;
        }
        galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type()); // Provide CPU backend buffer type
        if (!galloc) throw std::runtime_error("Failed to initialize GGML graph allocator");
        if (!ggml_gallocr_reserve(galloc, graph)) throw std::runtime_error("Failed to reserve memory for computation graph");
    }

    ggml_tensor* Florence2Processor::load_tensor(const std::string& name) {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("Tensor not found: " + name);
        return it->second;
    }

    int Florence2Processor::get_metadata_int(gguf_context* ctx, const std::string& key) {
        int idx = gguf_find_key(ctx, key.c_str());
        if (idx == -1) throw std::runtime_error("Metadata key not found: " + key);
        if (gguf_get_kv_type(ctx, idx) != GGUF_TYPE_UINT32) throw std::runtime_error("Metadata key " + key + " is not uint32");
        return gguf_get_val_u32(ctx, idx);
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
        if (image_tensor.dim() != 4 || image_tensor.size(0) != 1 || image_tensor.size(1) != 3 ||
            image_tensor.size(2) != 768 || image_tensor.size(3) != 768) {
            throw std::runtime_error("Unexpected image tensor shape");
        }
        image_tensor = image_tensor.permute({3, 2, 1, 0}).contiguous();

        std::vector<float> image_data(image_tensor.data_ptr<float>(), image_tensor.data_ptr<float>() + image_tensor.numel());
        size_t expected_size = 768 * 768 * 3 * 1;
        if (image_data.size() != expected_size) {
            throw std::runtime_error("Image data size mismatch: expected " + std::to_string(expected_size) +
                                     ", got " + std::to_string(image_data.size()));
        }

        std::vector<ggml_fp16_t> image_data_f16(expected_size);
        for (size_t i = 0; i < expected_size; ++i) image_data_f16[i] = ggml_fp32_to_fp16(image_data[i]);
        ggml_fp16_t* pixel_data = static_cast<ggml_fp16_t*>(ggml_get_data(pixel_values_prealloc));
        if (!pixel_data) throw std::runtime_error("pixel_values_prealloc data pointer is NULL");
        std::copy(image_data_f16.begin(), image_data_f16.end(), pixel_data);

        std::cout << "pixel_values_prealloc data[0]: " << pixel_data[0] << "\n";

        ggml_graph_clear(graph);
        reset_graph_allocator();

        // Copy GGUF tensors into ctx
        ggml_tensor* v_c0w_gguf = load_tensor("v_c0w");
        ggml_tensor* v_c0w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 7, 7, 3, 256);
        ggml_cpy(ctx, v_c0w_gguf, v_c0w);

        ggml_tensor* v_c0b_gguf = load_tensor("v_c0b");
        ggml_tensor* v_c0b = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 256);
        ggml_cpy(ctx, v_c0b_gguf, v_c0b);

        ggml_tensor* v_c1w_gguf = load_tensor("v_c1w");
        ggml_tensor* v_c1w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 256, 512);
        ggml_cpy(ctx, v_c1w_gguf, v_c1w);

        ggml_tensor* v_c1b_gguf = load_tensor("v_c1b");
        ggml_tensor* v_c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 512);
        ggml_cpy(ctx, v_c1b_gguf, v_c1b);

        ggml_tensor* v_c2w_gguf = load_tensor("v_c2w");
        ggml_tensor* v_c2w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 512, 1024);
        ggml_cpy(ctx, v_c2w_gguf, v_c2w);

        ggml_tensor* v_c2b_gguf = load_tensor("v_c2b");
        ggml_tensor* v_c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1024);
        ggml_cpy(ctx, v_c2b_gguf, v_c2b);

        ggml_tensor* v_pjw_gguf = load_tensor("v_pjw");
        int64_t dims_pjw[2] = { v_pjw_gguf->ne[0], v_pjw_gguf->ne[1] };
        ggml_tensor* v_pjw = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dims_pjw[0], dims_pjw[1]);
        ggml_cpy(ctx, v_pjw_gguf, v_pjw);

        ggml_tensor* t_out_w_gguf = load_tensor("t_out_w");
        int64_t dims_out_w[2] = { t_out_w_gguf->ne[0], t_out_w_gguf->ne[1] };
        ggml_tensor* t_out_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dims_out_w[0], dims_out_w[1]);
        ggml_cpy(ctx, t_out_w_gguf, t_out_w);

        // Build graph
        ggml_tensor* conv0 = ggml_conv_2d(ctx, v_c0w, pixel_values_prealloc, 4, 4, 3, 3, 1, 1);
        ggml_cpy(ctx, conv0, conv0_output);
        std::cout << "Allocated conv0\n";
        ggml_add_inplace(ctx, conv0_output, v_c0b);
        ggml_cpy(ctx, ggml_group_norm(ctx, conv0_output, 256, 1e-5f), norm0_prealloc);
        ggml_build_forward_expand(graph, norm0_prealloc);

        ggml_tensor* conv1 = ggml_conv_2d(ctx, v_c1w, norm0_prealloc, 2, 2, 1, 1, 1, 1);
        ggml_cpy(ctx, conv1, conv1_output);
        std::cout << "Allocated conv1\n";
        ggml_add_inplace(ctx, conv1_output, v_c1b);
        ggml_cpy(ctx, ggml_group_norm(ctx, conv1_output, 512, 1e-5f), norm1_prealloc);
        ggml_build_forward_expand(graph, norm1_prealloc);

        ggml_tensor* conv2 = ggml_conv_2d(ctx, v_c2w, norm1_prealloc, 2, 2, 1, 1, 1, 1);
        ggml_cpy(ctx, conv2, conv2_output);
        std::cout << "Allocated conv2\n";
        ggml_add_inplace(ctx, conv2_output, v_c2b);
        ggml_cpy(ctx, ggml_group_norm(ctx, conv2_output, 1024, 1e-5f), norm2_prealloc);
        ggml_build_forward_expand(graph, norm2_prealloc);

        ggml_tensor* vision_output = ggml_mul_mat(ctx, v_pjw, norm2_prealloc);
        ggml_build_forward_expand(graph, vision_output);

        int64_t* input_ids_data = static_cast<int64_t*>(ggml_get_data(input_ids_tensor_prealloc));
        std::copy(input_ids.begin(), input_ids.end(), input_ids_data);

        ggml_tensor* encoder_output = input_ids_tensor_prealloc;
        ggml_tensor* logits = ggml_mul_mat(ctx, t_out_w, encoder_output);
        ggml_build_forward_expand(graph, logits);

        if (!ggml_gallocr_alloc_graph(galloc, graph)) throw std::runtime_error("Failed to allocate graph memory");
        ggml_graph_compute(graph, NULL); // Corrected to 2 arguments

        ModelOutput output;
        output.logits.resize(vocab_size);
        float* logits_data = static_cast<float*>(ggml_get_data(logits));
        std::copy(logits_data, logits_data + vocab_size, output.logits.begin());
        output.past_key_values.resize(t_dec_layers);

        size_t used_mem = ggml_used_mem(ctx);
        std::cout << "Memory used after computation: " << used_mem << " bytes (~" << used_mem / (1024.0 * 1024 * 1024) << " GB)\n";

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
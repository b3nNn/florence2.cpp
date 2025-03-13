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

        vocab_size = get_metadata_int(temp_gguf_ctx, "vocab_size"); // 51289
        t_dec_layers = get_metadata_int(temp_gguf_ctx, "t_dec_layers"); // 12
        t_enc_layers = get_metadata_int(temp_gguf_ctx, "t_enc_layers"); // 12
        t_heads = get_metadata_int(temp_gguf_ctx, "t_heads"); // 16
        t_hidden = get_metadata_int(temp_gguf_ctx, "t_hidden"); // 1024
        t_ffn_dim = get_metadata_int(temp_gguf_ctx, "t_ffn_dim"); // 4096
        v_img_size = get_metadata_int(temp_gguf_ctx, "v_img_size"); // 768

        for (int i = 0; i < n_tensors; i++) {
            std::string name = gguf_get_tensor_name(temp_gguf_ctx, i);
            ggml_tensor* tensor = ggml_get_tensor(gguf_ctx, name.c_str());
            if (!tensor) throw std::runtime_error("Failed to get tensor: " + name);
            tensors[name] = tensor;
        }
        gguf_free(temp_gguf_ctx);
        gguf_free(file_ctx);

        size_t runtime_size = 0;
        runtime_size += 768 * 768 * 3 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;      // pixel_values_prealloc
        runtime_size += 192 * 192 * 256 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;    // conv0_output
        runtime_size += 192 * 192 * 256 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;    // conv0_bias
        runtime_size += 192 * 192 * 256 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;    // norm0_prealloc
        runtime_size += 96 * 96 * 512 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;      // conv1_output
        runtime_size += 96 * 96 * 512 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;      // conv1_bias
        runtime_size += 96 * 96 * 512 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;      // norm1_prealloc
        runtime_size += 48 * 48 * 1024 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;     // conv2_output
        runtime_size += 48 * 48 * 1024 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;     // conv2_bias
        runtime_size += 48 * 48 * 1024 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;     // norm2_prealloc
        runtime_size += 256 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;                // input_ids_tensor_prealloc
        runtime_size += 5 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;                  // decoder_input_ids_tensor_prealloc
        runtime_size += 192 * 192 * 256 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;    // conv0 intermediate
        runtime_size += 96 * 96 * 512 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;      // conv1 intermediate
        runtime_size += 48 * 48 * 1024 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;     // conv2 intermediate
        runtime_size += 7 * 7 * 3 * 256 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;        // v_c0w
        runtime_size += 1 * 1 * 256 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;        // v_c0b
        runtime_size += 3 * 3 * 256 * 512 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;      // v_c1w
        runtime_size += 1 * 1 * 512 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;        // v_c1b
        runtime_size += 3 * 3 * 512 * 1024 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;     // v_c2w
        runtime_size += 1 * 1 * 1024 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;       // v_c2b
        runtime_size += 1024 * 2048 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;            // v_pjw
        runtime_size += 51289 * 1024 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;           // t_out_w
        runtime_size += 51289 * 1024 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;           // t_wte (embedding)

        std::cout << "Runtime size: " << runtime_size << " bytes (~" << runtime_size / (1024.0 * 1024 * 1024) << " GB)\n";

        size_t graph_size = 1800ULL * 1024 * 1024; // 1.8 GB
        size_t total_size = runtime_size + graph_size + 1024ULL * 1024 * 1024; // 1 GB buffer
        total_size = GGML_PAD(total_size, GGML_MEM_ALIGN);

        ggml_init_params params_main = { .mem_size = total_size, .mem_buffer = NULL };
        ctx = ggml_init(params_main);
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
        input_ids_tensor_prealloc = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 256, 1); // Adjusted to I64 for input_ids
        decoder_input_ids_tensor_prealloc = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 5, 1);

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
        galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
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

    ggml_tensor* Florence2Processor::process_image(const cv::Mat& image) {
        torch::Tensor image_tensor = image_processor.Preprocess(image);
        if (image_tensor.dim() != 4 || image_tensor.size(0) != 1 || image_tensor.size(1) != 3 ||
            image_tensor.size(2) != 768 || image_tensor.size(3) != 768) {
            throw std::runtime_error("Unexpected image tensor shape");
        }
        image_tensor = image_tensor.permute({3, 2, 1, 0}).contiguous();

        std::vector<float> image_data(image_tensor.data_ptr<float>(), image_tensor.data_ptr<float>() + image_tensor.numel());
        size_t expected_size = 768 * 768 * 3 * 1;
        if (image_data.size() != expected_size) {
            throw std::runtime_error("Image data size mismatch");
        }

        std::vector<ggml_fp16_t> image_data_f16(expected_size);
        for (size_t i = 0; i < expected_size; ++i) image_data_f16[i] = ggml_fp32_to_fp16(image_data[i]);
        ggml_fp16_t* pixel_data = static_cast<ggml_fp16_t*>(ggml_get_data(pixel_values_prealloc));
        if (!pixel_data) throw std::runtime_error("pixel_values_prealloc data pointer is NULL");
        std::copy(image_data_f16.begin(), image_data_f16.end(), pixel_data);

        std::cout << "pixel_values_prealloc data[0]: " << ggml_fp16_to_fp32(pixel_data[0]) << "\n";

        ggml_tensor* v_c0w = load_tensor("v_c0w");
        ggml_tensor* v_c0b = load_tensor("v_c0b");
        ggml_tensor* v_c1w = load_tensor("v_c1w");
        ggml_tensor* v_c1b = load_tensor("v_c1b");
        ggml_tensor* v_c2w = load_tensor("v_c2w");
        ggml_tensor* v_c2b = load_tensor("v_c2b");
        ggml_tensor* v_pjw = load_tensor("v_pjw");

        // Reshape biases
        v_c0b = ggml_reshape_4d(ctx, v_c0b, 1, 1, 256, 1);
        v_c1b = ggml_reshape_4d(ctx, v_c1b, 1, 1, 512, 1);
        v_c2b = ggml_reshape_4d(ctx, v_c2b, 1, 1, 1024, 1);

        ggml_tensor* conv0 = ggml_conv_2d(ctx, v_c0w, pixel_values_prealloc, 4, 4, 3, 3, 1, 1);
        ggml_build_forward_expand(graph, conv0);
        std::cout << "conv0 shape: [" << conv0->ne[0] << ", " << conv0->ne[1] << ", "
                  << conv0->ne[2] << ", " << conv0->ne[3] << "]\n";
        ggml_tensor* conv0_with_bias = ggml_add(ctx, conv0, v_c0b);
        ggml_build_forward_expand(graph, conv0_with_bias);
        ggml_tensor* norm0 = ggml_group_norm(ctx, conv0_with_bias, 256, 1e-5f);
        ggml_build_forward_expand(graph, norm0);

        ggml_tensor* conv1 = ggml_conv_2d(ctx, v_c1w, norm0, 2, 2, 1, 1, 1, 1);
        ggml_build_forward_expand(graph, conv1);
        std::cout << "conv1 shape: [" << conv1->ne[0] << ", " << conv1->ne[1] << ", "
                  << conv1->ne[2] << ", " << conv1->ne[3] << "]\n";
        ggml_tensor* conv1_with_bias = ggml_add(ctx, conv1, v_c1b);
        ggml_build_forward_expand(graph, conv1_with_bias);
        ggml_tensor* norm1 = ggml_group_norm(ctx, conv1_with_bias, 512, 1e-5f);
        ggml_build_forward_expand(graph, norm1);

        ggml_tensor* conv2 = ggml_conv_2d(ctx, v_c2w, norm1, 2, 2, 1, 1, 1, 1);
        ggml_build_forward_expand(graph, conv2);
        std::cout << "conv2 shape: [" << conv2->ne[0] << ", " << conv2->ne[1] << ", "
                  << conv2->ne[2] << ", " << conv2->ne[3] << "]\n";
        ggml_tensor* conv2_with_bias = ggml_add(ctx, conv2, v_c2b);
        ggml_build_forward_expand(graph, conv2_with_bias);
        ggml_tensor* norm2 = ggml_group_norm(ctx, conv2_with_bias, 1024, 1e-5f);
        ggml_build_forward_expand(graph, norm2);
        std::cout << "norm2 shape: [" << norm2->ne[0] << ", " << norm2->ne[1] << ", "
                  << norm2->ne[2] << ", " << norm2->ne[3] << "]\n";

        ggml_tensor* norm2_reshaped = ggml_reshape_2d(ctx, norm2, 48 * 48, 1024);
        ggml_build_forward_expand(graph, norm2_reshaped);
        std::cout << "norm2_reshaped shape: [" << norm2_reshaped->ne[0] << ", " << norm2_reshaped->ne[1] << "]\n";
        ggml_tensor* norm2_pooled = ggml_mean(ctx, norm2_reshaped); // [1, 1024]
        ggml_build_forward_expand(graph, norm2_pooled);
        std::cout << "norm2_pooled shape: [" << norm2_pooled->ne[0] << ", " << norm2_pooled->ne[1] << "]\n";

        // Reshape norm2_pooled to [1024, 1] for ggml_mul_mat
        ggml_tensor* norm2_pooled_reshaped = ggml_reshape_2d(ctx, norm2_pooled, 1024, 1);
        ggml_build_forward_expand(graph, norm2_pooled_reshaped);
        std::cout << "norm2_pooled_reshaped shape: [" << norm2_pooled_reshaped->ne[0] << ", " << norm2_pooled_reshaped->ne[1] << "]\n";

        ggml_tensor* vision_output = ggml_mul_mat(ctx, v_pjw, norm2_pooled_reshaped); // [1024, 1024] * [1024, 1] = [1024, 1]
        if (!vision_output) {
            throw std::runtime_error("Vision output GGML matrix multiplication failed");
        }
        ggml_build_forward_expand(graph, vision_output);
        std::cout << "vision_output shape: [" << vision_output->ne[0] << ", " << vision_output->ne[1] << "]\n";

        return vision_output;
    }

    ModelOutput Florence2Processor::process(const std::string& text, const cv::Mat& image, const std::vector<int64_t>& decoder_input_ids,
                                            const std::vector<PastKeyValueLayer>& past_key_values) {
        std::cout << "Processing text: " << text << "\n";
        std::string prompt = task_prompts.count(text) ? task_prompts.at(text) : text;
        std::vector<uint32_t> token_ids = tokenizer.encode(prompt, true, 256 - image_seq_length, true, true, "longest");
        std::vector<int64_t> input_ids(token_ids.begin(), token_ids.end());
        input_ids.resize(256, 0);

        ggml_graph_clear(graph);
        reset_graph_allocator();

        ggml_tensor* vision_output = process_image(image); // [1, 1024]

        // Embed text input_ids
        int64_t* input_ids_data = static_cast<int64_t*>(ggml_get_data(input_ids_tensor_prealloc));
        std::copy(input_ids.begin(), input_ids.end(), input_ids_data);
        ggml_tensor* t_wte = load_tensor("t_wte"); // [51289, 1024] embedding matrix
        ggml_tensor* text_embeddings = ggml_mul_mat(ctx, t_wte, input_ids_tensor_prealloc); // [51289, 1024] * [256, 1] = [256, 1024]
        ggml_build_forward_expand(graph, text_embeddings);
        std::cout << "text_embeddings shape: [" << text_embeddings->ne[0] << ", " << text_embeddings->ne[1] << "]\n";

        // Concatenate vision and text embeddings
        ggml_tensor* encoder_input = ggml_concat(ctx, vision_output, text_embeddings, 0); // [1+256, 1024] = [257, 1024]
        ggml_build_forward_expand(graph, encoder_input);
        std::cout << "encoder_input shape: [" << encoder_input->ne[0] << ", " << encoder_input->ne[1] << "]\n";

        // Placeholder for decoder (simplified to logits generation)
        ggml_tensor* t_out_w = load_tensor("t_out_w"); // [1024, 51289]
        ggml_tensor* logits = ggml_mul_mat(ctx, t_out_w, encoder_input); // [1024, 51289] * [257, 1024] = [257, 51289]
        ggml_build_forward_expand(graph, logits);
        std::cout << "logits shape: [" << logits->ne[0] << ", " << logits->ne[1] << "]\n";

        if (!ggml_gallocr_alloc_graph(galloc, graph)) throw std::runtime_error("Failed to allocate graph memory");
        ggml_graph_compute(graph, NULL);

        ModelOutput output;
        output.logits.resize(vocab_size);
        ggml_fp16_t* logits_data = static_cast<ggml_fp16_t*>(ggml_get_data(logits));
        for (size_t i = 0; i < vocab_size; ++i) {
            output.logits[i] = ggml_fp16_to_fp32(logits_data[i]); // Take first token’s logits for simplicity
        }
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
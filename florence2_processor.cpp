#include "florence2_processor.hpp"
#include <filesystem>
#include <cassert>

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
        assert(gguf_ctx != nullptr && "Failed to initialize GGUF context");

        gguf_init_params temp_params = { .no_alloc = true, .ctx = NULL };
        gguf_context* temp_gguf_ctx = gguf_init_from_file(gguf_path.c_str(), temp_params);
        assert(temp_gguf_ctx != nullptr && "Failed to initialize temporary GGUF context");

        gguf_init_params load_params = { .no_alloc = false, .ctx = &gguf_ctx };
        gguf_context* file_ctx = gguf_init_from_file(gguf_path.c_str(), load_params);
        assert(file_ctx != nullptr && "Failed to load GGUF file into gguf_ctx");

        vocab_size = get_metadata_int(temp_gguf_ctx, "vocab_size");
        t_dec_layers = get_metadata_int(temp_gguf_ctx, "t_dec_layers");
        t_enc_layers = get_metadata_int(temp_gguf_ctx, "t_enc_layers");
        t_heads = get_metadata_int(temp_gguf_ctx, "t_heads");
        t_hidden = get_metadata_int(temp_gguf_ctx, "t_hidden");
        t_ffn_dim = get_metadata_int(temp_gguf_ctx, "t_ffn_dim");
        v_img_size = get_metadata_int(temp_gguf_ctx, "v_img_size");

        for (int i = 0; i < gguf_get_n_tensors(temp_gguf_ctx); i++) {
            std::string name = gguf_get_tensor_name(temp_gguf_ctx, i);
            ggml_tensor* tensor = ggml_get_tensor(gguf_ctx, name.c_str());
            assert(tensor != nullptr && ("Failed to get tensor: " + name).c_str());
            tensors[name] = tensor;
        }
        gguf_free(temp_gguf_ctx);
        gguf_free(file_ctx);

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
        runtime_size += 1 * 1 * 256 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 3 * 3 * 256 * 512 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 1 * 1 * 512 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 3 * 3 * 512 * 1024 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 1 * 1 * 1024 * 1 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 1024 * 2048 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;
        runtime_size += 51289 * 1024 * sizeof(ggml_fp16_t) + GGML_TENSOR_SIZE;

        size_t graph_size = 1800ULL * 1024 * 1024;
        size_t total_size = runtime_size + graph_size + 1024ULL * 1024 * 1024;
        total_size = GGML_PAD(total_size, GGML_MEM_ALIGN);

        ggml_init_params params_main = { .mem_size = total_size, .mem_buffer = NULL };
        ctx = ggml_init(params_main);
        assert(ctx != nullptr && "Failed to initialize main GGML context");

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

        graph = ggml_new_graph(ctx);
        assert(graph != nullptr && "Failed to initialize computation graph");

        reset_graph_allocator();
    }

    void Florence2Processor::reset_graph_allocator() {
        if (galloc) ggml_gallocr_free(galloc);
        galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        assert(galloc != nullptr && "Failed to initialize GGML graph allocator");
        assert(ggml_gallocr_reserve(galloc, graph) && "Failed to reserve memory for computation graph");
    }

    ggml_tensor* Florence2Processor::load_tensor(const std::string& name) {
        auto it = tensors.find(name);
        assert(it != tensors.end() && ("Tensor not found: " + name).c_str());
        return it->second;
    }

    int Florence2Processor::get_metadata_int(gguf_context* ctx, const std::string& key) {
        int idx = gguf_find_key(ctx, key.c_str());
        assert(idx != -1 && ("Metadata key not found: " + key).c_str());
        assert(gguf_get_kv_type(ctx, idx) == GGUF_TYPE_UINT32 && ("Metadata key " + key + " is not uint32").c_str());
        return gguf_get_val_u32(ctx, idx);
    }

    // Preprocess image into GGML tensor
    ggml_tensor* preprocess_image(ggml_context* ctx, ggml_tensor* pixel_values_prealloc, const cv::Mat& image, CLIPImageProcessor& image_processor) {
        torch::Tensor image_tensor = image_processor.Preprocess(image);
        assert(image_tensor.dim() == 4 && image_tensor.size(0) == 1 && image_tensor.size(1) == 3 &&
               image_tensor.size(2) == 768 && image_tensor.size(3) == 768 && "Unexpected image tensor shape");
        image_tensor = image_tensor.permute({3, 2, 1, 0}).contiguous();

        size_t expected_size = 768 * 768 * 3 * 1;
        std::vector<float> image_data(image_tensor.data_ptr<float>(), image_tensor.data_ptr<float>() + image_tensor.numel());
        assert(image_data.size() == expected_size && "Image data size mismatch");

        std::vector<ggml_fp16_t> image_data_f16(expected_size);
        for (size_t i = 0; i < expected_size; ++i) image_data_f16[i] = ggml_fp32_to_fp16(image_data[i]);
        ggml_fp16_t* pixel_data = static_cast<ggml_fp16_t*>(ggml_get_data(pixel_values_prealloc));
        assert(pixel_data != nullptr && "pixel_values_prealloc data pointer is NULL");
        std::copy(image_data_f16.begin(), image_data_f16.end(), pixel_data);

        return pixel_values_prealloc;
    }

    // Apply convolutional stages
    ggml_tensor* apply_conv_stages(ggml_context* ctx, ggml_cgraph* graph, ggml_tensor* input,
                                   ggml_tensor* v_c0w, ggml_tensor* v_c0b,
                                   ggml_tensor* v_c1w, ggml_tensor* v_c1b,
                                   ggml_tensor* v_c2w, ggml_tensor* v_c2b) {
        v_c0b = ggml_reshape_4d(ctx, v_c0b, 1, 1, 256, 1);
        ggml_tensor* conv0 = ggml_conv_2d(ctx, v_c0w, input, 4, 4, 3, 3, 1, 1);
        ggml_build_forward_expand(graph, conv0);
        ggml_tensor* conv0_with_bias = ggml_add(ctx, conv0, v_c0b);
        ggml_build_forward_expand(graph, conv0_with_bias);
        ggml_tensor* norm0 = ggml_group_norm(ctx, conv0_with_bias, 256, 1e-5f);
        ggml_build_forward_expand(graph, norm0);

        v_c1b = ggml_reshape_4d(ctx, v_c1b, 1, 1, 512, 1);
        ggml_tensor* conv1 = ggml_conv_2d(ctx, v_c1w, norm0, 2, 2, 1, 1, 1, 1);
        ggml_build_forward_expand(graph, conv1);
        ggml_tensor* conv1_with_bias = ggml_add(ctx, conv1, v_c1b);
        ggml_build_forward_expand(graph, conv1_with_bias);
        ggml_tensor* norm1 = ggml_group_norm(ctx, conv1_with_bias, 512, 1e-5f);
        ggml_build_forward_expand(graph, norm1);

        v_c2b = ggml_reshape_4d(ctx, v_c2b, 1, 1, 1024, 1);
        ggml_tensor* conv2 = ggml_conv_2d(ctx, v_c2w, norm1, 2, 2, 1, 1, 1, 1);
        ggml_build_forward_expand(graph, conv2);
        ggml_tensor* conv2_with_bias = ggml_add(ctx, conv2, v_c2b);
        ggml_build_forward_expand(graph, conv2_with_bias);
        ggml_tensor* norm2 = ggml_group_norm(ctx, conv2_with_bias, 1024, 1e-5f);
        ggml_build_forward_expand(graph, norm2);

        return norm2;
    }

    // Pool spatial dimensions
    ggml_tensor* pool_spatial(ggml_context* ctx, ggml_cgraph* graph, ggml_tensor* norm2) {
        ggml_tensor* reshaped = ggml_reshape_2d(ctx, norm2, 48 * 48, 1024);
        ggml_build_forward_expand(graph, reshaped);
        ggml_tensor* pooled = ggml_mean(ctx, reshaped);
        ggml_build_forward_expand(graph, pooled);
        return pooled;
    }

    // Project to text embedding space
    ggml_tensor* project_vision(ggml_context* ctx, ggml_cgraph* graph, ggml_tensor* pooled) {
        ggml_tensor* transposed = ggml_transpose(ctx, pooled);
        ggml_build_forward_expand(graph, transposed);

        ggml_tensor* projection = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1024, 1024);
        for (int i = 0; i < 1024 * 1024; ++i) {
            ((ggml_fp16_t*)projection->data)[i] = ggml_fp32_to_fp16(i % 1024 == i / 1024 ? 1.0f : 0.0f);
        }
        ggml_tensor* output = ggml_mul_mat(ctx, projection, transposed);
        assert(output != nullptr && "Vision output GGML matrix multiplication failed");
        ggml_build_forward_expand(graph, output);

        ggml_tensor* corrected = ggml_reshape_2d(ctx, output, 1, 1024);
        ggml_build_forward_expand(graph, corrected);
        return corrected;
    }

    ggml_tensor* Florence2Processor::process_image(const cv::Mat& image) {
        ggml_tensor* input = preprocess_image(ctx, pixel_values_prealloc, image, image_processor);
        ggml_tensor* features = apply_conv_stages(ctx, graph, input,
                                                  load_tensor("v_c0w"), load_tensor("v_c0b"),
                                                  load_tensor("v_c1w"), load_tensor("v_c1b"),
                                                  load_tensor("v_c2w"), load_tensor("v_c2b"));
        ggml_tensor* pooled = pool_spatial(ctx, graph, features);
        return project_vision(ctx, graph, pooled);
    }

    ModelOutput Florence2Processor::process(const std::string& text, const cv::Mat& image, const std::vector<int64_t>& decoder_input_ids,
                                            const std::vector<PastKeyValueLayer>& past_key_values) {
        std::string prompt = task_prompts.count(text) ? task_prompts.at(text) : text;
        std::vector<uint32_t> token_ids = tokenizer.encode(prompt, true, 256 - image_seq_length, true, true, "longest");
        std::vector<int64_t> input_ids(token_ids.begin(), token_ids.end());
        input_ids.resize(256, 0);

        ggml_graph_clear(graph);
        reset_graph_allocator();

        ggml_tensor* vision_output = process_image(image);
        int64_t* input_ids_data = static_cast<int64_t*>(ggml_get_data(input_ids_tensor_prealloc));
        std::copy(input_ids.begin(), input_ids.end(), input_ids_data);
        ggml_tensor* t_wte = load_tensor("t_wte");
        ggml_tensor* text_embeddings = ggml_mul_mat(ctx, t_wte, input_ids_tensor_prealloc);
        ggml_build_forward_expand(graph, text_embeddings);

        ggml_tensor* encoder_input = ggml_concat(ctx, vision_output, text_embeddings, 0);
        ggml_build_forward_expand(graph, encoder_input);

        ggml_tensor* t_out_w = load_tensor("t_out_w");
        ggml_tensor* logits = ggml_mul_mat(ctx, t_out_w, encoder_input);
        ggml_build_forward_expand(graph, logits);

        assert(ggml_gallocr_alloc_graph(galloc, graph) && "Failed to allocate graph memory");
        ggml_graph_compute(graph, NULL);

        ModelOutput output;
        output.logits.resize(vocab_size);
        ggml_fp16_t* logits_data = static_cast<ggml_fp16_t*>(ggml_get_data(logits));
        for (size_t i = 0; i < vocab_size; ++i) {
            output.logits[i] = ggml_fp16_to_fp32(logits_data[i]);
        }
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
            if (token_id == 2) break;
        }
        return decode(token_ids, true);
    }

} // namespace Florence2Processor
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

        // Main context
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

    // Manual matrix multiplication helper
    ggml_tensor* Florence2Processor::manual_mul_mat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
        int rows_a = a->ne[1]; // Rows of a
        int cols_a = a->ne[0]; // Columns of a
        int rows_b = b->ne[1]; // Rows of b
        int cols_b = b->ne[0]; // Columns of b

        if (cols_a != rows_b) {
            std::cerr << "Matrix multiplication mismatch: " << cols_a << " != " << rows_b << "\n";
            return nullptr;
        }

        ggml_tensor* result = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cols_b, rows_a); // [rows_a, cols_b]
        ggml_fp16_t* a_data = (ggml_fp16_t*)a->data;
        ggml_fp16_t* b_data = (ggml_fp16_t*)b->data;
        ggml_fp16_t* result_data = (ggml_fp16_t*)result->data;

        for (int i = 0; i < rows_a; ++i) { // Rows of result
            for (int j = 0; j < cols_b; ++j) { // Columns of result
                float sum = 0.0f;
                for (int k = 0; k < cols_a; ++k) { // Columns of a, rows of b
                    sum += ggml_fp16_to_fp32(a_data[i * cols_a + k]) * ggml_fp16_to_fp32(b_data[k * cols_b + j]);
                }
                result_data[i * cols_b + j] = ggml_fp32_to_fp16(sum);
            }
        }

        return result;
    }

    ggml_tensor* manual_conv_2d(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight,
                                int s_w, int s_h, int p_w, int p_h, int d_w, int d_h) {
        int in_w = input->ne[0], in_h = input->ne[1], in_c = input->ne[2];
        int k_w = weight->ne[0], k_h = weight->ne[1], k_in_c = weight->ne[2], k_out_c = weight->ne[3];

        int out_w = (in_w + 2 * p_w - k_w) / s_w + 1; // 14
        int out_h = (in_h + 2 * p_h - k_h) / s_h + 1; // 14
        int out_c = k_out_c; // 4

        ggml_tensor* output = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, out_w, out_h, out_c, 1);
        ggml_fp16_t* in_data = (ggml_fp16_t*)input->data;
        ggml_fp16_t* w_data = (ggml_fp16_t*)weight->data;
        ggml_fp16_t* out_data = (ggml_fp16_t*)output->data;

        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                for (int oc = 0; oc < out_c; ++oc) {
                    float sum = 0.0f;
                    for (int ky = 0; ky < k_h; ++ky) {
                        int iy = oy * s_h + ky - p_h;
                        if (iy < 0 || iy >= in_h) continue;
                        for (int kx = 0; kx < k_w; ++kx) {
                            int ix = ox * s_w + kx - p_w;
                            if (ix < 0 || ix >= in_w) continue;
                            for (int ic = 0; ic < in_c; ++ic) {
                                float in_val = ggml_fp16_to_fp32(in_data[iy * in_w * in_c + ix * in_c + ic]);
                                float w_val = ggml_fp16_to_fp32(w_data[ky * k_w * k_in_c * k_out_c + kx * k_in_c * k_out_c + ic * k_out_c + oc]);
                                sum += in_val * w_val;
                            }
                        }
                    }
                    int idx = oy * out_w * out_c + ox * out_c + oc;
                    out_data[idx] = ggml_fp32_to_fp16(sum);
                }
            }
        }

        return output;
    }

    ggml_tensor* manual_add(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
        int w = a->ne[0], h = a->ne[1], c = a->ne[2], n = a->ne[3];
        ggml_tensor* result = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, w, h, c, n);
        ggml_fp16_t* a_data = (ggml_fp16_t*)a->data;
        ggml_fp16_t* b_data = (ggml_fp16_t*)b->data;
        ggml_fp16_t* r_data = (ggml_fp16_t*)result->data;

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                for (int k = 0; k < c; ++k) {
                    int idx = i * w * c + j * c + k;
                    float a_val = ggml_fp16_to_fp32(a_data[idx]);
                    float b_val = ggml_fp16_to_fp32(b_data[k]);
                    r_data[idx] = ggml_fp32_to_fp16(a_val + b_val);
                }
            }
        }

        return result;
    }

    // Extracted image processing logic
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

        ggml_tensor* conv0 = manual_conv_2d(ctx, pixel_values_prealloc, v_c0w, 4, 4, 3, 3, 1, 1);
        std::cout << "conv0 shape: [" << conv0->ne[0] << ", " << conv0->ne[1] << ", "
                  << conv0->ne[2] << ", " << conv0->ne[3] << "]\n";
        ggml_tensor* conv0_with_bias = manual_add(ctx, conv0, v_c0b);
        ggml_tensor* norm0 = ggml_group_norm(ctx, conv0_with_bias, 256, 1e-5f);

        ggml_tensor* conv1 = manual_conv_2d(ctx, norm0, v_c1w, 2, 2, 1, 1, 1, 1);
        std::cout << "conv1 shape: [" << conv1->ne[0] << ", " << conv1->ne[1] << ", "
                  << conv1->ne[2] << ", " << conv1->ne[3] << "]\n";
        ggml_tensor* conv1_with_bias = manual_add(ctx, conv1, v_c1b);
        ggml_tensor* norm1 = ggml_group_norm(ctx, conv1_with_bias, 512, 1e-5f);

        ggml_tensor* conv2 = manual_conv_2d(ctx, norm1, v_c2w, 2, 2, 1, 1, 1, 1);
        std::cout << "conv2 shape: [" << conv2->ne[0] << ", " << conv2->ne[1] << ", "
                  << conv2->ne[2] << ", " << conv2->ne[3] << "]\n";
        ggml_tensor* conv2_with_bias = manual_add(ctx, conv2, v_c2b);
        ggml_tensor* norm2 = ggml_group_norm(ctx, conv2_with_bias, 1024, 1e-5f);
        std::cout << "norm2 shape: [" << norm2->ne[0] << ", " << norm2->ne[1] << ", "
                  << norm2->ne[2] << ", " << norm2->ne[3] << "]\n";

        ggml_tensor* norm2_reshaped = ggml_reshape_2d(ctx, norm2, 48 * 48, 1024);
        std::cout << "norm2_reshaped shape: [" << norm2_reshaped->ne[0] << ", " << norm2_reshaped->ne[1] << "]\n";
        ggml_tensor* norm2_pooled_temp = ggml_mean(ctx, norm2_reshaped);
        std::cout << "norm2_pooled_temp shape: [" << norm2_pooled_temp->ne[0] << ", " << norm2_pooled_temp->ne[1] << "]\n";

        ggml_tensor* temp_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1024, 2048);
        for (int i = 0; i < 1024 * 2048; ++i) {
            ((ggml_fp16_t*)temp_weight->data)[i] = ggml_fp32_to_fp16(i % 1024 == i / 2048 ? 1.0f : 0.0f);
        }
        ggml_tensor* norm2_transformed = manual_mul_mat(ctx, temp_weight, ggml_transpose(ctx, norm2_pooled_temp));
        std::cout << "norm2_transformed shape: [" << norm2_transformed->ne[0] << ", " << norm2_transformed->ne[1] << "]\n";
        ggml_tensor* norm2_pooled = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 1);
        ggml_cpy(ctx, norm2_transformed, norm2_pooled);
        std::cout << "norm2_pooled shape: [" << norm2_pooled->ne[0] << ", " << norm2_pooled->ne[1] << "]\n";

        ggml_tensor* vision_output = manual_mul_mat(ctx, v_pjw, norm2_pooled);
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

        // Process image
        ggml_tensor* vision_output = process_image(image);
        ggml_build_forward_expand(graph, vision_output);

        // Process text
        int64_t* input_ids_data = static_cast<int64_t*>(ggml_get_data(input_ids_tensor_prealloc));
        std::copy(input_ids.begin(), input_ids.end(), input_ids_data);

        ggml_tensor* encoder_output = input_ids_tensor_prealloc; // [256, 1]
        ggml_tensor* t_out_w = load_tensor("t_out_w"); // [1024, 51289]
        ggml_tensor* logits = manual_mul_mat(ctx, t_out_w, encoder_output); // [51289, 1]
        std::cout << "logits shape: [" << logits->ne[0] << ", " << logits->ne[1] << "]\n";
        ggml_build_forward_expand(graph, logits);

        if (!ggml_gallocr_alloc_graph(galloc, graph)) throw std::runtime_error("Failed to allocate graph memory");
        ggml_graph_compute(graph, NULL);

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
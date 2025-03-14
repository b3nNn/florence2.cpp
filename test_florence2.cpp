#include "florence2_processor.hpp"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

//int main() {
//    try {
//        std::string gguf_path = "florence2.gguf";
//        std::string tokenizer_path = "tokenizer.json";
//        std::string preprocessor_config_path = "preprocessor_config.json";
//        std::string tensors_mapping_path = "tensors_mapping.json";
//        if (!std::filesystem::exists("puma.png")) {
//            std::cerr << "puma.png not found in current directory\n";
//            return 1;
//        }
//        Florence2Processor::Florence2Processor processor(gguf_path, tokenizer_path, preprocessor_config_path, tensors_mapping_path);
//        cv::Mat image = cv::imread("puma.png");
//        if (image.empty()) {
//            std::cerr << "Failed to load puma.png—check file integrity\n";
//            return 1;
//        }
//        std::string caption = processor.generate("<CAPTION>", image, 5);
//        std::cout << "Caption: " << caption << "\n";
//    } catch (const std::exception& e) {
//        std::cerr << "Exception: " << e.what() << "\n";
//        return 1;
//    }
//    return 0;
//}

#include <ggml.h>
#include <ggml-alloc.h>
#include <iostream>
#include <cassert>

int main() {
    // Initialize GGML backend (CPU)
    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr && "Failed to initialize CPU backend");

    // Initialize GGML context with backend
    size_t mem_size = 1024 * 1024 * 1024; // 1GB buffer
    ggml_init_params params = { .mem_size = mem_size, .mem_buffer = nullptr, .no_alloc = true };
    ggml_context* ctx = ggml_init(params);
    assert(ctx != nullptr && "Failed to initialize GGML context");

    // Create computation graph
    ggml_cgraph* graph = ggml_new_graph(ctx);
    assert(graph != nullptr && "Failed to initialize graph");

    // Initialize allocator with backend buffer type
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(alloc != nullptr && "Failed to initialize allocator");

    // Correct tensor shapes
    ggml_tensor* vision_projection_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1024, 2048, 1, 1); // [1024, 2048, 1, 1]
    ggml_tensor* pooled = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 2048, 1, 1, 1); // [2048, 1, 1, 1]

    // Define multiplication in the graph
    ggml_tensor* output = ggml_mul_mat(ctx, vision_projection_weight, pooled); // [1024, 2048] * [2048, 1] = [1024, 1]
    ggml_build_forward_expand(graph, output);

    // Optionally reserve buffers for the graph (recommended to avoid reallocations)
    ggml_gallocr_reserve(alloc, graph);

    // Allocate memory for the graph (input, intermediate, and output tensors)
    assert(ggml_gallocr_alloc_graph(alloc, graph) && "Failed to allocate graph");

    // Explicitly initialize tensor data (required for input tensors after allocation)
    ggml_set_zero(vision_projection_weight); // Initialize vision_projection_weight to zero
    ggml_set_zero(pooled); // Initialize pooled to zero

    // Log shapes
    std::cout << "vision_projection_weight shape: [" << vision_projection_weight->ne[0] << ", " << vision_projection_weight->ne[1] << ", " << vision_projection_weight->ne[2] << ", " << vision_projection_weight->ne[3] << "]\n";
    std::cout << "pooled shape: [" << pooled->ne[0] << ", " << pooled->ne[1] << ", " << pooled->ne[2] << ", " << pooled->ne[3] << "]\n";
    std::cout << "output shape (pre-compute): [" << output->ne[0] << ", " << output->ne[1] << ", " << output->ne[2] << ", " << output->ne[3] << "]\n";

    // Compute the graph
    ggml_graph_compute(graph, nullptr);

    std::cout << "output shape (post-compute): [" << output->ne[0] << ", " << output->ne[1] << ", " << output->ne[2] << ", " << output->ne[3] << "]\n";

    // Validate
    if (output->ne[0] == 1024 && output->ne[1] == 1 && output->ne[2] == 1 && output->ne[3] == 1) {
        std::cout << "PoC Passed: ggml_mul_mat produced expected shape [1024, 1, 1, 1]\n";
    } else {
        std::cout << "PoC Failed: Unexpected output shape\n";
    }

    // Cleanup
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
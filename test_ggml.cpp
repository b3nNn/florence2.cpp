#include <ggml.h>
#include <ggml-backend.h>
#include <iostream>
#include <cstring>

int main() {
    std::cout << "Starting GGML Matrix Multiplication Example\n";

    // Initialize CPU backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::cerr << "Failed to initialize CPU backend\n";
        return 1;
    }
    std::cout << "Backend initialized: OK\n";

    // Initialize context
    size_t mem_size = 1024 * 1024 * 1024; // 1GB
    struct ggml_init_params params = { .mem_size = mem_size, .mem_buffer = nullptr, .no_alloc = false };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to initialize GGML context\n";
        ggml_backend_free(backend);
        return 1;
    }
    std::cout << "Context initialized: OK\n";

    // Create computation graph
    ggml_cgraph* graph = ggml_new_graph(ctx);
    if (!graph) {
        std::cerr << "Failed to initialize graph\n";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    std::cout << "Graph created: OK\n";

    ggml_tensor* vision_projection_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1024);
    float* vision_data = new float[1024 * 2048];
    std::fill_n(vision_data, 1024 * 2048, 1.0f);
    memcpy(vision_projection_weight->data, vision_data, 1024 * 2048 * sizeof(float));
    std::cout << "vision_projection_weight shape: [" << vision_projection_weight->ne[0] << ", "
              << vision_projection_weight->ne[1] << ", " << vision_projection_weight->ne[2] << ", "
              << vision_projection_weight->ne[3] << "]\n";

    ggml_tensor* pooled = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    float* pooled_data = new float[2048];
    std::fill_n(pooled_data, 2048, 1.0f);
    memcpy(pooled->data, pooled_data, 2048 * sizeof(float));
    std::cout << "pooled shape: [" << pooled->ne[0] << ", " << pooled->ne[1] << ", "
              << pooled->ne[2] << ", " << pooled->ne[3] << "]\n";

    // Perform matrix multiplication
    std::cout << "Performing ggml_mul_mat...\n";
    ggml_tensor* result = ggml_mul_mat(ctx, vision_projection_weight, pooled);
    if (!result) {
        std::cerr << "Failed to perform ggml_mul_mat\n";
        delete[] vision_data;
        delete[] pooled_data;
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    std::cout << "Result shape: [" << result->ne[0] << ", " << result->ne[1] << ", "
              << result->ne[2] << ", " << result->ne[3] << "]\n";

    // Build and compute the graph
    ggml_build_forward_expand(graph, result);
    struct ggml_cplan cplan = ggml_graph_plan(graph, 1, NULL);
    uint8_t* work_buffer = nullptr;
    if (cplan.work_size > 0) {
        work_buffer = static_cast<uint8_t*>(malloc(cplan.work_size));
        if (!work_buffer) {
            std::cerr << "Failed to allocate work buffer\n";
            delete[] vision_data;
            delete[] pooled_data;
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 1;
        }
        cplan.work_data = work_buffer;
    }
    if (ggml_graph_compute(graph, &cplan) != GGML_STATUS_SUCCESS) {
        std::cerr << "Failed to compute graph\n";
        delete[] vision_data;
        delete[] pooled_data;
        if (work_buffer) free(work_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    std::cout << "Graph computation completed\n";

    // Verify result
    float* result_data = (float*)result->data;
    std::cout << "Sample result[0]: " << result_data[0] << " (expected 2048.0)\n";

    // Clean up
    delete[] vision_data;
    delete[] pooled_data;
    if (work_buffer) free(work_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::cout << "Cleanup completed\n";
    return 0;
}
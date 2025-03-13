#include <ggml.h>
#include <gguf.h>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Helper to convert GGML type to string
std::string ggml_type_to_string(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16: return "float16";
        case GGML_TYPE_F32: return "float32";
        case GGML_TYPE_I8: return "int8";
        case GGML_TYPE_I16: return "int16";
        case GGML_TYPE_I32: return "int32";
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_Q4_1: return "q4_1";
        default: return "unknown";
    }
}

// Load tensor mapping from JSON
std::map<std::string, std::string> load_tensor_mapping(const std::string& mapping_path) {
    std::map<std::string, std::string> gguf_to_pytorch;
    std::ifstream file(mapping_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << mapping_path << ". Proceeding without mapping.\n";
        return gguf_to_pytorch;
    }

    json mapping;
    file >> mapping;
    for (auto& [pytorch_name, gguf_name] : mapping.items()) {
        gguf_to_pytorch[gguf_name] = pytorch_name; // Invert key<->value from mapping
    }
    return gguf_to_pytorch;
}

// Print tensor details with mapping
void print_tensor_details(ggml_tensor* tensor, const std::string& name,
                          const std::map<std::string, std::string>& mapping) {
    std::cout << "Tensor: " << name << "\n";
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        std::cout << "  Full Name: " << it->second << "\n";
    } else {
        std::cout << "  Full Name: <not mapped>\n";
    }
    std::cout << "  Shape: [" << tensor->ne[0] << ", " << tensor->ne[1] << ", "
              << tensor->ne[2] << ", " << tensor->ne[3] << "]\n";
    std::cout << "  Type: " << ggml_type_to_string(tensor->type) << "\n";
    std::cout << "  Elements: " << ggml_nelements(tensor) << "\n\n";
}

// Main GGUF inspection function
void inspect_gguf(const std::string& gguf_path, const std::string& mapping_path) {
    // Load tensor mapping
    std::map<std::string, std::string> tensor_mapping = load_tensor_mapping(mapping_path);

    // Initialize GGUF context without allocation
    gguf_init_params params = { .no_alloc = true, .ctx = nullptr };
    gguf_context* gguf_ctx = gguf_init_from_file(gguf_path.c_str(), params);
    assert(gguf_ctx != nullptr && "Failed to load GGUF file");

    // Print metadata
    std::cout << "=== Metadata ===\n";
    int n_kv = gguf_get_n_kv(gguf_ctx);
    for (int i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(gguf_ctx, i);
        enum gguf_type type = gguf_get_kv_type(gguf_ctx, i);

        std::cout << key << ": ";
        switch (type) {
            case GGUF_TYPE_UINT32:
                std::cout << gguf_get_val_u32(gguf_ctx, i);
                break;
            case GGUF_TYPE_INT32:
                std::cout << gguf_get_val_i32(gguf_ctx, i);
                break;
            case GGUF_TYPE_FLOAT32:
                std::cout << gguf_get_val_f32(gguf_ctx, i);
                break;
            case GGUF_TYPE_STRING: {
                const char* val = gguf_get_val_str(gguf_ctx, i);
                std::cout << val;
                break;
            }
            default:
                std::cout << "<unsupported type>";
                break;
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Initialize GGML context to load tensors
    size_t gguf_tensor_size = gguf_get_tensor_offset(gguf_ctx, gguf_get_n_tensors(gguf_ctx) - 1) +
                              ggml_tensor_overhead() * gguf_get_n_tensors(gguf_ctx);
    ggml_init_params ggml_params = { .mem_size = gguf_tensor_size + 1024 * 1024, .mem_buffer = nullptr };
    ggml_context* ctx = ggml_init(ggml_params);
    assert(ctx != nullptr && "Failed to initialize GGML context");

    gguf_init_params load_params = { .no_alloc = false, .ctx = &ctx };
    gguf_context* file_ctx = gguf_init_from_file(gguf_path.c_str(), load_params);
    assert(file_ctx != nullptr && "Failed to load GGUF file into GGML context");

    // Print tensor details
    std::cout << "=== Tensors ===\n";
    int n_tensors = gguf_get_n_tensors(file_ctx);
    for (int i = 0; i < n_tensors; ++i) {
        std::string name = gguf_get_tensor_name(file_ctx, i);
        ggml_tensor* tensor = ggml_get_tensor(ctx, name.c_str());
        assert(tensor != nullptr && ("Failed to get tensor: " + name).c_str());
        print_tensor_details(tensor, name, tensor_mapping);
    }

    // Cleanup
    gguf_free(file_ctx);
    ggml_free(ctx);
    gguf_free(gguf_ctx);
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <gguf_file_path> [<mapping_file_path>]\n";
        return 1;
    }

    std::string gguf_path = argv[1];
    std::string mapping_path = (argc == 3) ? argv[2] : "tensors_mapping.json";

    try {
        inspect_gguf(gguf_path, mapping_path);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
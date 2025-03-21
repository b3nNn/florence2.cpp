#include <torch/torch.h>
#include <memory>
#include <vector>
#include <tuple>

// Define the PreNorm module as a template class
template <typename NormType, typename FnType>
struct PreNorm : torch::nn::Module {
    // Constructor
    PreNorm(NormType norm, FnType fn, std::shared_ptr<torch::nn::Module> drop_path = nullptr) {
        // Register the normalization module (can be nullptr)
        this->norm = register_module("norm", std::move(norm));
        // Register the function module
        this->fn = register_module("fn", std::move(fn));
        // Register drop_path if provided
        if (drop_path) {
            this->drop_path = register_module("drop_path", drop_path);
        } else {
            this->drop_path = nullptr;
        }
    }

    // Forward method with variadic arguments
    template <typename... Args>
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, Args... args) {
        // Store the input for the shortcut connection
        torch::Tensor shortcut = x;

        // Apply normalization if norm is not null
        torch::Tensor input_to_fn;
        if (!norm.isNone()) {
            input_to_fn = norm->forward(x);
        } else {
            input_to_fn = x;
        }

        // Prepare inputs for fn, including the normalized input and additional arguments
        std::vector<c10::IValue> fn_inputs = {input_to_fn};
        (fn_inputs.push_back(args), ...);  // Fold expression to add variadic args

        // Call the function module and expect a tuple of two tensors (x, size)
        auto fn_output = fn->forward(fn_inputs);
        auto outputs = fn_output.toTuple();
        torch::Tensor new_x = outputs->elements()[0].toTensor();
        torch::Tensor size = outputs->elements()[1].toTensor();

        // Apply drop_path if it exists
        if (drop_path != nullptr) {
            new_x = drop_path->forward(new_x);
        }

        // Add the shortcut connection
        new_x = shortcut + new_x;

        // Return the updated tensor and size
        return std::make_tuple(new_x, size);
    }

    // Member variables
    NormType norm{nullptr};  // Normalization module (e.g., LayerNorm)
    FnType fn{nullptr};      // Function module (e.g., attention or feed-forward)
    std::shared_ptr<torch::nn::Module> drop_path{nullptr};  // Optional drop path module
};
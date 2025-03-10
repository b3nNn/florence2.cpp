#include "florence2_processor.hpp"
#include <stdexcept>
#include <vector>
#include <memory>

namespace Florence2Processor {

    Florence2Processor::Florence2Processor(const std::string& model_path,
                                           const std::string& tokenizer_json_path,
                                           const std::string& image_config_path)
            : env(ORT_LOGGING_LEVEL_WARNING, "Florence2"),
              session(nullptr),
              tokenizer(tokenizer_json_path),
              image_processor(image_config_path) {
        // Initialize ONNX Runtime session
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        try {
            session = Ort::Session(env, model_path.c_str(), session_options);
        } catch (const Ort::Exception& e) {
            throw std::runtime_error(std::string("Failed to load ONNX model: ") + e.what());
        }
    }

    std::vector<float> Florence2Processor::process(const std::string& text, const cv::Mat& image) {
        // Tokenize text
        std::vector<uint32_t> token_ids = tokenizer.encode(text, true, 256, true, true, "longest");
        std::vector<int64_t> input_ids(token_ids.begin(), token_ids.end());
        std::vector<int64_t> attention_mask(token_ids.size(), 1);
        input_ids.resize(256, 0);  // Pad to 256
        attention_mask.resize(256, 0);

        // Dummy decoder_input_ids (BOS token = 0, padded to 256)
        std::vector<int64_t> decoder_input_ids(256, 0);  // Adjust BOS if tokenizer specifies (e.g., 2)

        // Preprocess image
        torch::Tensor image_tensor = image_processor.Preprocess(image);  // Assumes [0, 1] output, [1, 3, 224, 224]
        if (image_tensor.sizes() != torch::IntArrayRef({1, 3, 224, 224})) {
            throw std::runtime_error("Image tensor shape mismatch, expected [1, 3, 224, 224]");
        }
        std::vector<float> image_data(image_tensor.data_ptr<float>(), image_tensor.data_ptr<float>() + image_tensor.numel());

        // Prepare ONNX inputs
        Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;

        // pixel_values: [1, 3, 224, 224]
        std::vector<int64_t> pixel_shape = {1, 3, 224, 224};
        inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info, image_data.data(), image_data.size(), pixel_shape.data(), pixel_shape.size()));

        // input_ids: [1, 256]
        std::vector<int64_t> input_shape = {1, 256};
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size()));

        // attention_mask: [1, 256]
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size()));

        // decoder_input_ids: [1, 256]
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, decoder_input_ids.data(), decoder_input_ids.size(), input_shape.data(), input_shape.size()));

        // Define input and output names matching the ONNX model
        const char* input_names[] = {"pixel_values", "input_ids", "attention_mask", "decoder_input_ids"};
        const char* output_names[] = {"logits"};

        // Run inference
        std::vector<Ort::Value> output_tensors;
        try {
            output_tensors = session.Run(Ort::RunOptions{}, input_names, inputs.data(), inputs.size(), output_names, 1);
        } catch (const Ort::Exception& e) {
            throw std::runtime_error(std::string("Inference failed: ") + e.what());
        }

        // Extract logits
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t output_size = 1;
        for (auto dim : output_shape) output_size *= dim;

        return std::vector<float>(output_data, output_data + output_size);
    }

} // namespace Florence2Processor
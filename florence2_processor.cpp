#include "florence2_processor.hpp"
#include <stdexcept>
#include <windows.h>

namespace Florence2Processor {

    std::wstring string_to_wstring(const std::string& str) {
        if (str.empty()) return std::wstring();
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
        std::wstring wstr(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], size_needed);
        return wstr;
    }

    Florence2Processor::Florence2Processor(const std::string& model_path,
                                           const std::string& tokenizer_json_path,
                                           const std::string& image_config_path)
            : env(ORT_LOGGING_LEVEL_WARNING, "Florence2"),
              session(nullptr),
              tokenizer(tokenizer_json_path),
              image_processor(image_config_path, 768) {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        std::wstring w_model_path = string_to_wstring(model_path);
        session = Ort::Session(env, w_model_path.c_str(), session_options);
    }

    std::vector<float> Florence2Processor::process(const std::string& text, const cv::Mat& image) {
        std::vector<uint32_t> token_ids = tokenizer.encode(text, true, 256, true, true, "longest");
        std::vector<int64_t> input_ids(token_ids.begin(), token_ids.end());
        input_ids.resize(256, 0);
        std::vector<int64_t> decoder_input_ids(256, 0);

        torch::Tensor image_tensor = image_processor.Preprocess(image);
        if (image_tensor.sizes() != torch::IntArrayRef({1, 3, 768, 768})) {
            throw std::runtime_error("Image tensor shape mismatch, expected [1, 3, 768, 768]");
        }
        std::vector<float> image_data(image_tensor.data_ptr<float>(), image_tensor.data_ptr<float>() + image_tensor.numel());

        Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        std::vector<int64_t> pixel_shape = {1, 3, 768, 768};
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info, image_data.data(), image_data.size(), pixel_shape.data(), pixel_shape.size()));
        std::vector<int64_t> input_shape = {1, 256};
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, decoder_input_ids.data(), decoder_input_ids.size(), input_shape.data(), input_shape.size()));

        const char* input_names[] = {"pixel_values", "input_ids", "decoder_input_ids"};
        const char* output_names[] = {"logits"};
        std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{}, input_names, inputs.data(), 3, output_names, 1);
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t output_size = 1;
        for (auto dim : output_shape) output_size *= dim;

        return std::vector<float>(output_data, output_data + output_size);
    }

    std::string Florence2Processor::decode(const std::vector<uint32_t>& token_ids, bool skip_special_tokens) {
        return tokenizer.decode(token_ids, skip_special_tokens);
    }

} // namespace Florence2Processor
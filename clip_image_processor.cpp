#include "clip_image_processor.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

CLIPImageProcessor::CLIPImageProcessor(const std::string& config_path, int target_size)
        : target_size_(target_size) {
    // Load config if needed, but we’ll hard-code CLIP defaults for simplicity
    std::ifstream file(config_path);
    nlohmann::json config;
    file >> config;
    // Could parse mean/std if needed, but using CLIP defaults here
}

torch::Tensor CLIPImageProcessor::Preprocess(const cv::Mat& image) {
    // Resize image to target_size_ (768x768)
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(target_size_, target_size_), 0, 0, cv::INTER_LINEAR);

    // Convert cv::Mat (BGR) to torch::Tensor (RGB)
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
    torch::Tensor tensor = torch::from_blob(resized_image.data, {resized_image.rows, resized_image.cols, 3}, torch::kUInt8);
    tensor = tensor.permute({2, 0, 1}).to(torch::kFloat32) / 255.0;  // [3, 768, 768]

    // Normalize with CLIP defaults
    torch::Tensor mean = torch::tensor({0.48145466, 0.4578275, 0.40821073}).view({3, 1, 1});
    torch::Tensor std = torch::tensor({0.26862954, 0.26130258, 0.27577711}).view({3, 1, 1});
    tensor = (tensor - mean) / std;

    return tensor.unsqueeze(0);  // [1, 3, 768, 768]
}
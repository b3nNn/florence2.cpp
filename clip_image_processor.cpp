#include "clip_image_processor.hpp"
#include <nlohmann/json.hpp> // Add this dependency for JSON parsing
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

CLIPImageProcessor::CLIPImageProcessor(const std::string& config_path) {
    load_config(config_path);
}

void CLIPImageProcessor::load_config(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_path);
    }

    json config;
    file >> config;

    // Extract configuration values with defaults if not present
    do_resize = config.value("do_resize", true);
    do_normalize = config.value("do_normalize", true);
    size = config.value("size", 224); // Default size if not specified
    mean = config.value("mean", std::vector<float>{0.48145466, 0.4578275, 0.40821073});
    std = config.value("std", std::vector<float>{0.26862954, 0.26130258, 0.27577711});
    swapRB = config.value("swapRB", false);
    name = config.value("name", "clip_vit");
    do_center_crop = config.value("do_center_crop", true);
    crop_size = config.value("crop_size", 224);

    // Validate extracted values
    if (mean.size() != 3 || std.size() != 3) {
        throw std::runtime_error("Mean and std must be vectors of size 3 in config: " + config_path);
    }
    if (size <= 0 || crop_size <= 0) {
        throw std::runtime_error("Size and crop_size must be positive in config: " + config_path);
    }
}

torch::Tensor CLIPImageProcessor::Preprocess(const cv::Mat& image) {
    // Existing preprocessing logic (unchanged)
    cv::Mat processed_image = image.clone();

    // Swap RB channels if required
    if (swapRB) {
        cv::cvtColor(processed_image, processed_image, cv::COLOR_BGR2RGB);
    }

    // Center crop if required
    if (do_center_crop) {
        int h = processed_image.rows;
        int w = processed_image.cols;
        int min_dim = std::min(h, w);
        int crop_y = (h - min_dim) / 2;
        int crop_x = (w - min_dim) / 2;
        processed_image = processed_image(cv::Rect(crop_x, crop_y, min_dim, min_dim));
        cv::resize(processed_image, processed_image, cv::Size(crop_size, crop_size));
    }

    // Resize if required
    if (do_resize && (processed_image.rows != size || processed_image.cols != size)) {
        cv::resize(processed_image, processed_image, cv::Size(size, size));
    }

    // Convert to tensor
    torch::Tensor tensor_image = torch::from_blob(processed_image.data, {processed_image.rows, processed_image.cols, 3}, torch::kByte);
    tensor_image = tensor_image.permute({2, 0, 1}).to(torch::kFloat) / 255.0;

    // Normalize if required
    if (do_normalize) {
        tensor_image = (tensor_image - torch::tensor(mean).view({3, 1, 1})) / torch::tensor(std).view({3, 1, 1});
    }

    // Add batch dimension
    return tensor_image.unsqueeze(0);
}
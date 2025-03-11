#include "florence2_processor.hpp"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <random>
#include <torch/torch.h>

int main() {
    try {
        Florence2Processor::Florence2Processor processor(
            "florence2.onnx",
            "tokenizer.json",
            "preprocessor_config.json"
        );

        cv::Mat image = cv::imread("puma.png");
        if (image.empty()) {
            throw std::runtime_error("Failed to load image 'puma.png'");
        }

        std::string text = "<CAPTION>";
        std::string caption = processor.generate(text, image, 20);

        std::cout << "Generated caption: " << caption << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
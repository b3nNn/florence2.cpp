#include "florence2_processor.hpp"
#include <iostream>
#include <filesystem>
#include <torch/torch.h>

int main(int argc, char** argv) {
    try {
        Florence2Processor::Florence2Processor processor(
                "florence2.onnx",
                "tokenizer.json",
                "config.json"
        );

        cv::Mat image = cv::imread("puma.png");
        if (image.empty()) {
            throw std::runtime_error("Failed to load image 'puma.png'");
        }

        std::string text = "<CAPTION>";
        std::vector<float> output = processor.process(text, image);

        std::cout << "Inference output size: " << output.size() << std::endl;
        std::cout << "First few logits: ";
        for (size_t i = 0; i < std::min<size_t>(5, output.size()); ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
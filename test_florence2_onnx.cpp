#include "florence2_processor.hpp"
#include <iostream>
#include <filesystem>
#include <torch/torch.h>

int main(int argc, char** argv) {
    try {
        Florence2Processor::Florence2Processor processor(
                "pytorch_model.bin",
                "tokenizer.json",
                "preprocessor_config.json"
        );

        cv::Mat image = cv::imread("example_image.jpg");
        if (image.empty()) {
            throw std::runtime_error("Could not load image");
        }

        std::string text = "What is in the image?";
        auto output = processor.process(text, image);

        std::cout << "Output shape: " << output.sizes() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
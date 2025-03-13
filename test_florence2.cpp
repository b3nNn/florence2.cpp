#include "florence2_processor.hpp"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    try {
        std::string gguf_path = "florence2.gguf";
        std::string tokenizer_path = "tokenizer.json";
        std::string image_config_path = "preprocessor_config.json";
        if (!std::filesystem::exists("puma.png")) {
            std::cerr << "puma.png not found in current directory\n";
            return 1;
        }
        Florence2Processor::Florence2Processor processor(gguf_path, tokenizer_path, image_config_path);
        cv::Mat image = cv::imread("puma.png");
        if (image.empty()) {
            std::cerr << "Failed to load puma.png—check file integrity\n";
            return 1;
        }
        std::string caption = processor.generate("<CAPTION>", image, 5);
        std::cout << "Caption: " << caption << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
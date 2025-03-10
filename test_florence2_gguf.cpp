#include "florence2_processor.hpp"
#include <iostream>
#include <filesystem>
#include <torch/torch.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }

    std::string model_path = std::filesystem::absolute(argv[1]).string();
    std::string image_path = std::filesystem::absolute(argv[2]).string();
    std::cerr << "Resolved model path: " << model_path << std::endl;
    std::cerr << "Resolved image path: " << image_path << std::endl;

    Florence2Processor processor(
            model_path,
            "tokenizer.json",
            "",
            {224, 224}
    );

    std::cerr << "Processor constructed" << std::endl;

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }

    auto input = processor.preprocess("", image, "caption");
    auto output_ids = processor.generate(input, 50);
    auto output = processor.postprocess(output_ids, "caption", image);

    std::cout << "Caption: " << output.text << std::endl;
    return 0;
}
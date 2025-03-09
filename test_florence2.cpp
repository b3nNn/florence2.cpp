#include "florence2_processor.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

int main() {
    try {
        Florence2Processor processor("tokenizer.json"); // Adjust path

        cv::Mat image = cv::imread("puma.png", cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image");
        }
        std::cout << "Original image size: [" << image.cols << ", " << image.rows << "]\n";

        auto input = processor.preprocess("", image, "caption");
        std::cout << "Input IDs: ";
        for (uint32_t id : input.input_ids) {
            std::cout << id << " ";
        }
        std::cout << "\nPixel values shape: ["
                  << input.pixel_values.size() << ", "
                  << input.pixel_values[0].channels() << ", "
                  << input.pixel_values[0].rows << ", "
                  << input.pixel_values[0].cols << "]\n";

        std::cout << "First few values (channel 0, top-left corner):\n";
        std::vector<cv::Mat> channels;
        cv::split(input.pixel_values[0], channels);
        for (int i = 0; i < 5 && i < channels[0].rows; ++i) {
            for (int j = 0; j < 5 && j < channels[0].cols; ++j) {
                std::cout << channels[0].at<float>(i, j) << " ";
            }
            std::cout << "\n";
        }

        std::cout << "Mean across channels: [";
        std::vector<double> means;
        for (int c = 0; c < input.pixel_values[0].channels(); ++c) {
            cv::Scalar mean_std = cv::mean(channels[c]);
            means.push_back(mean_std[0]);
            std::cout << mean_std[0] << (c < 2 ? " " : "");
        }
        std::cout << "]\nStd across channels: [";
        for (int c = 0; c < input.pixel_values[0].channels(); ++c) {
            cv::Mat diff;
            cv::subtract(channels[c], means[c], diff);
            cv::Mat squared_diff;
            cv::multiply(diff, diff, squared_diff);
            cv::Scalar mean_squared_diff = cv::mean(squared_diff);
            double std = std::sqrt(mean_squared_diff[0]);
            std::cout << std << (c < 2 ? " " : "");
        }
        std::cout << "]\n";

        std::vector<uint32_t> output_ids = {0, 250, 2721, 18820, 81, 5, 6444, 4, 2};
        auto output = processor.postprocess(output_ids, "caption");
        std::cout << "Caption: " << output.text << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 0;
}
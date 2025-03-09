#include "clip_image_processor.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    try {
        CLIPImageProcessor processor;
        cv::Mat image = cv::imread("puma.png", cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image");
        }
        std::cout << "Original image size: [" << image.cols << ", " << image.rows << "]" << std::endl;

        auto processed = processor(image);
        cv::Mat& pixel_values = processed.pixel_values;

        // Print shape (simulate batch dimension of 1)
        std::cout << "Processed image shape: [1, "
                  << pixel_values.channels() << ", "
                  << pixel_values.rows << ", "
                  << pixel_values.cols << "]" << std::endl;

        // Print first few values (channel 0, top-left 5x5)
        std::cout << "First few values (channel 0, top-left corner):" << std::endl;
        std::vector<cv::Mat> channels;
        cv::split(pixel_values, channels);
        for (int i = 0; i < 5 && i < channels[0].rows; ++i) {
            for (int j = 0; j < 5 && j < channels[0].cols; ++j) {
                std::cout << channels[0].at<float>(i, j) << " ";
            }
            std::cout << std::endl;
        }

        // Compute mean and std across channels
        for (int c = 0; c < pixel_values.channels(); ++c) {
            cv::Scalar mean_std = cv::mean(channels[c]);
            double mean = mean_std[0];
            cv::Mat mean_diff;
            cv::absdiff(channels[c], mean, mean_diff);
            cv::Scalar std = cv::mean(mean_diff);
            std::cout << "Channel " << c << " - Mean: " << mean << ", Std: " << std << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
#include "clip_image_processor.hpp"

CLIPImageProcessor::CLIPImageProcessor(bool do_resize, Size size, int interpolation,
                                       bool do_center_crop, Size crop_size,
                                       bool do_normalize, double scale,
                                       bool do_convert_rgb, std::vector<double> mean, std::vector<double> std)
        : do_resize_(do_resize), size_(size), interpolation_(interpolation),
          do_center_crop_(do_center_crop), crop_size_(crop_size),
          do_normalize_(do_normalize), scale_(scale),
          do_convert_rgb_(do_convert_rgb), mean_(mean), std_(std) {}

cv::Mat CLIPImageProcessor::preprocess(const cv::Mat& image) {
    cv::Mat processed = image.clone();
    if (do_resize_) {
        cv::resize(processed, processed, cv::Size(size_.width, size_.height), 0, 0, interpolation_);
    }
    if (do_center_crop_) {
        int x = (processed.cols - crop_size_.width) / 2;
        int y = (processed.rows - crop_size_.height) / 2;
        processed = processed(cv::Rect(x, y, crop_size_.width, crop_size_.height));
    }
    if (do_convert_rgb_) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    }
    if (do_normalize_) {
        processed.convertTo(processed, CV_32F, scale_);
        cv::subtract(processed, cv::Scalar(mean_[0], mean_[1], mean_[2]), processed);
        cv::divide(processed, cv::Scalar(std_[0], std_[1], std_[2]), processed);
    }
    return processed;
}

std::vector<cv::Mat> CLIPImageProcessor::preprocess(const std::vector<cv::Mat>& images) {
    std::vector<cv::Mat> processed_images;
    processed_images.reserve(images.size());
    for (const auto& image : images) {
        processed_images.push_back(preprocess(image));
    }
    return processed_images;
}
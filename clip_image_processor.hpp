#ifndef CLIP_IMAGE_PROCESSOR_HPP
#define CLIP_IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class CLIPImageProcessor {
public:
    struct Size {
        int height;
        int width;
    };

    struct ProcessedOutput {
        cv::Mat pixel_values; // Shape: [batch_size, channels, height, width]
    };

    CLIPImageProcessor(
            bool do_resize = true,
            Size size = {336, 336}, // Update default to 336x336 for Florence-2
            int resample = cv::INTER_LINEAR,
            bool do_center_crop = true,
            Size crop_size = {336, 336}, // Update default to 336x336
            bool do_rescale = true,
            double rescale_factor = 1.0 / 255.0,
            bool do_normalize = true,
            std::vector<double> image_mean = {0.48145466, 0.4578275, 0.40821073},
            std::vector<double> image_std = {0.26862954, 0.26130258, 0.27577711}
    );

    ProcessedOutput operator()(const cv::Mat& image);
    ProcessedOutput operator()(const std::vector<cv::Mat>& images);

private:
    bool do_resize_;
    Size size_;
    int resample_;
    bool do_center_crop_;
    Size crop_size_;
    bool do_rescale_;
    double rescale_factor_;
    bool do_normalize_;
    std::vector<double> image_mean_;
    std::vector<double> image_std_;

    cv::Mat preprocess(const cv::Mat& image);
    cv::Mat resize(const cv::Mat& image);
    cv::Mat center_crop(const cv::Mat& image);
    cv::Mat rescale(const cv::Mat& image);
    cv::Mat normalize(const cv::Mat& image);
};

#endif // CLIP_IMAGE_PROCESSOR_HPP
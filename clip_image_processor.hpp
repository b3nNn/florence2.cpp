#ifndef CLIP_IMAGE_PROCESSOR_HPP
#define CLIP_IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class CLIPImageProcessor {
public:
    struct Size {
        int height, width;
    };

    CLIPImageProcessor(bool do_resize, Size size, int interpolation,
                       bool do_center_crop, Size crop_size,
                       bool do_normalize, double scale,
                       bool do_convert_rgb, std::vector<double> mean, std::vector<double> std);

    cv::Mat preprocess(const cv::Mat& image);  // Made public
    std::vector<cv::Mat> preprocess(const std::vector<cv::Mat>& images);  // New overload

private:
    bool do_resize_;
    Size size_;
    int interpolation_;
    bool do_center_crop_;
    Size crop_size_;
    bool do_normalize_;
    double scale_;
    bool do_convert_rgb_;
    std::vector<double> mean_;
    std::vector<double> std_;

    // Private helper methods (if any) remain private
};

#endif
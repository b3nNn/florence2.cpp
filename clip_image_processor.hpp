#ifndef CLIP_IMAGE_PROCESSOR_HPP
#define CLIP_IMAGE_PROCESSOR_HPP

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>

class CLIPImageProcessor {
public:
    CLIPImageProcessor(const std::string& config_path, int target_size = 768);
    torch::Tensor Preprocess(const cv::Mat& image);

private:
    int target_size_;
};

#endif // CLIP_IMAGE_PROCESSOR_HPP
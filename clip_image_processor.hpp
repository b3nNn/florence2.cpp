#ifndef CLIP_IMAGE_PROCESSOR_HPP
#define CLIP_IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <string>
#include <vector>

class CLIPImageProcessor {
public:
    // Constructor takes only the config path
    CLIPImageProcessor(const std::string& config_path);

    // Preprocess an image and return a tensor
    torch::Tensor Preprocess(const cv::Mat& image);

private:
    // Configuration variables (populated from JSON config)
    bool do_resize;
    bool do_normalize;
    int size;
    std::vector<float> mean;
    std::vector<float> std;
    bool swapRB;
    std::string name;
    bool do_center_crop;
    int crop_size;

    // Helper method to load configuration from JSON file
    void load_config(const std::string& config_path);
};

#endif // CLIP_IMAGE_PROCESSOR_HPP
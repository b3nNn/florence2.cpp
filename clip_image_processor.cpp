#include "clip_image_processor.hpp"
#include <stdexcept>

CLIPImageProcessor::CLIPImageProcessor(
        bool do_resize, Size size, int resample, bool do_center_crop, Size crop_size,
        bool do_rescale, double rescale_factor, bool do_normalize,
        std::vector<double> image_mean, std::vector<double> image_std
) : do_resize_(do_resize), size_(size), resample_(resample),
    do_center_crop_(do_center_crop), crop_size_(crop_size),
    do_rescale_(do_rescale), rescale_factor_(rescale_factor),
    do_normalize_(do_normalize), image_mean_(image_mean), image_std_(image_std) {
    if (image_mean_.size() != 3 || image_std_.size() != 3) {
        throw std::invalid_argument("image_mean and image_std must have 3 elements (RGB)");
    }
}

CLIPImageProcessor::ProcessedOutput CLIPImageProcessor::operator()(const cv::Mat& image) {
    cv::Mat processed = preprocess(image);
    return ProcessedOutput{processed}; // Shape: [3, 224, 224], batch=1 implied in test output
}

CLIPImageProcessor::ProcessedOutput CLIPImageProcessor::operator()(const std::vector<cv::Mat>& images) {
    if (images.empty()) {
        throw std::invalid_argument("Input image batch is empty");
    }

    std::vector<cv::Mat> processed_images;
    processed_images.reserve(images.size());
    for (const auto& image : images) {
        processed_images.push_back(preprocess(image));
    }

    // Stack into a 4D cv::Mat: [batch_size, channels, height, width]
    cv::Mat batched;
    if (!processed_images.empty()) {
        int batch_size = static_cast<int>(processed_images.size());
        int channels = processed_images[0].channels();
        int height = processed_images[0].rows;
        int width = processed_images[0].cols;
        batched = cv::Mat(batch_size * channels, height * width, CV_32F); // Line 52
        for (int i = 0; i < batch_size; ++i) {
            std::vector<cv::Mat> channel_vec; // Renamed to avoid shadowing
            cv::split(processed_images[i], channel_vec);
            for (int c = 0; c < static_cast<int>(channel_vec.size()); ++c) {
                cv::Mat channel_flat = channel_vec[c].reshape(1, height * width);
                channel_flat.copyTo(batched.row(i * channels + c));
            }
        }
        batched = batched.reshape(1, {batch_size, channels, height, width});
    }

    return ProcessedOutput{batched};
}

cv::Mat CLIPImageProcessor::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    cv::Mat processed = image.clone();
    if (processed.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    }

    if (do_resize_) {
        processed = resize(processed);
    }
    if (do_center_crop_) {
        processed = center_crop(processed);
    }
    if (do_rescale_) {
        processed = rescale(processed);
    }
    if (do_normalize_) {
        processed = normalize(processed);
    }

    processed.convertTo(processed, CV_32F);
    return processed;
}

cv::Mat CLIPImageProcessor::resize(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(size_.width, size_.height), 0, 0, resample_);
    return resized;
}

cv::Mat CLIPImageProcessor::center_crop(const cv::Mat& image) {
    int height = image.rows;
    int width = image.cols;
    int crop_height = crop_size_.height;
    int crop_width = crop_size_.width;

    if (height < crop_height || width < crop_width) {
        throw std::runtime_error("Image too small for center crop");
    }

    int y = (height - crop_height) / 2;
    int x = (width - crop_width) / 2;
    cv::Rect roi(x, y, crop_width, crop_height);
    return image(roi).clone();
}

cv::Mat CLIPImageProcessor::rescale(const cv::Mat& image) {
    cv::Mat rescaled;
    image.convertTo(rescaled, CV_32F, rescale_factor_);
    return rescaled;
}

cv::Mat CLIPImageProcessor::normalize(const cv::Mat& image) {
    cv::Mat normalized = image.clone();
    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);

    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - image_mean_[i]) / image_std_[i];
    }

    cv::merge(channels, normalized);
    return normalized;
}
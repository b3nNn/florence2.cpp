#ifndef FLORENCE2_PROCESSOR_HPP
#define FLORENCE2_PROCESSOR_HPP

#include "bart_tokenizer_fast.hpp"
#include "clip_image_processor.hpp"
#include <unordered_map>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class Florence2Processor {
public:
    struct ProcessedInput {
        std::vector<uint32_t> input_ids;
        std::vector<cv::Mat> pixel_values;
        std::vector<uint32_t> attention_mask;
    };

    struct BoundingBox {
        std::vector<float> bbox;
        std::string label;
        float score;
    };

    struct ProcessedOutput {
        std::string text;
        std::vector<BoundingBox> bboxes;
        std::vector<std::string> labels;
    };

    Florence2Processor(
            const std::string& tokenizer_config_path = "path/to/tokenizer_config",
            const std::string& image_processor_config_path = "",
            CLIPImageProcessor::Size image_size = {336, 336} // Explicitly pass 336x336
    );

    ProcessedInput preprocess(
            const std::string& text_prompt,
            const cv::Mat& image,
            const std::string& task_prompt = "<CAPTION>"
    );
    ProcessedInput preprocess(
            const std::string& text_prompt,
            const std::vector<cv::Mat>& images,
            const std::string& task_prompt = "<CAPTION>"
    );

    ProcessedOutput postprocess(
            const std::vector<uint32_t>& output_ids,
            const std::string& task_prompt,
            const cv::Mat& image = cv::Mat()
    );

    BartTokenizerFast& get_tokenizer() { return tokenizer_; } // For testing

private:
    BartTokenizerFast tokenizer_;
    CLIPImageProcessor image_processor_;
    std::unordered_map<std::string, std::string> task_prompts_;

    void initialize_task_prompts();
    std::string get_task_prompt(const std::string& task_prompt);
};

#endif // FLORENCE2_PROCESSOR_HPP
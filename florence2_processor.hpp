#ifndef FLORENCE2_PROCESSOR_HPP
#define FLORENCE2_PROCESSOR_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "hf_tokenizer/hf_tokenizers.hpp"
#include "clip_image_processor.hpp"
#include <torch/torch.h>
#include <memory>

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
    };

    Florence2Processor(
            const std::string& model_path,
            const std::string& tokenizer_config_path,
            const std::string& image_processor_config_path,
            CLIPImageProcessor::Size image_size
    );
    ~Florence2Processor() = default;

    ProcessedInput preprocess(const std::string& text_prompt, const cv::Mat& image, const std::string& task_prompt);
    ProcessedInput preprocess(const std::string& text_prompt, const std::vector<cv::Mat>& images, const std::string& task_prompt);
    std::vector<uint32_t> generate(const ProcessedInput& input, int max_new_tokens);
    ProcessedOutput postprocess(const std::vector<uint32_t>& output_ids, const std::string& task_prompt, const cv::Mat& image);

private:
    struct Florence2ModelImpl : torch::nn::Module {
        torch::nn::Conv2d conv1;  // No nullptr, initialize in constructor
        torch::nn::Embedding embed_tokens;
        torch::nn::Linear img_proj;

        Florence2ModelImpl();  // Define in .cpp

        torch::Tensor forward(torch::Tensor pixel_values, torch::Tensor input_ids, torch::Tensor attention_mask,
                              torch::Tensor decoder_input_ids, torch::Tensor decoder_attention_mask) {
            auto vision_out = conv1->forward(pixel_values).view({1, -1});
            auto proj_out = img_proj->forward(vision_out);
            auto text_out = embed_tokens->forward(decoder_input_ids);
            return torch::cat({proj_out, text_out}, 1);
        }
    };

    using Florence2Model = torch::nn::ModuleHolder<Florence2ModelImpl>;

    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<CLIPImageProcessor> image_processor_;
    Florence2Model model_;

    std::unordered_map<std::string, std::string> task_prompts_;

    void initialize_task_prompts();
    std::string get_task_prompt(const std::string& task_prompt);
};

#endif
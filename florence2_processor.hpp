#ifndef FLORENCE2_PROCESSOR_HPP
#define FLORENCE2_PROCESSOR_HPP

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "bart_tokenizer_fast.hpp"
#include "clip_image_processor.hpp"

namespace Florence2Processor {

    class Florence2Processor {
    public:
        Florence2Processor(const std::string& model_path,
                           const std::string& tokenizer_json_path,
                           const std::string& image_config_path);
        std::vector<float> process(const std::string& text, const cv::Mat& image);
        std::string decode(const std::vector<uint32_t>& token_ids, bool skip_special_tokens = true);

    private:
        Ort::Env env;
        Ort::Session session;
        BartTokenizerFast tokenizer;
        CLIPImageProcessor image_processor;
    };

} // namespace Florence2Processor

#endif // FLORENCE2_PROCESSOR_HPP
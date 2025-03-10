#include "florence2_processor.hpp"
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <filesystem>

Florence2Processor::Florence2ModelImpl::Florence2ModelImpl()
        : conv1(torch::nn::Conv2dOptions(3, 256, 7).stride(4).padding(2)),
          embed_tokens(51289, 1024),
          img_proj(torch::nn::LinearOptions(2048, 1024)) {
    std::cerr << "Initializing Florence2ModelImpl" << std::endl;
    try {
        std::cerr << "Conv2d constructed" << std::endl;
        if (!conv1) std::cerr << "conv1 is null" << std::endl;

        std::cerr << "Embedding constructed" << std::endl;
        if (!embed_tokens) std::cerr << "embed_tokens is null" << std::endl;

        std::cerr << "Linear constructed" << std::endl;
        if (!img_proj) std::cerr << "img_proj is null" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception in Florence2ModelImpl init: " << e.what() << std::endl;
        throw;
    }
    std::cerr << "Florence2ModelImpl initialized" << std::endl;
}

Florence2Processor::Florence2Processor(
        const std::string& model_path,
        const std::string& tokenizer_config_path,
        const std::string& image_processor_config_path,
        CLIPImageProcessor::Size image_size
) : model_(Florence2ModelImpl()) {
    std::cerr << "Entering constructor" << std::endl;

    try {
        tokenizer_ = std::make_unique<Tokenizer>(tokenizer_config_path);
        std::cerr << "Tokenizer initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Tokenizer init failed: " << e.what() << std::endl;
        throw;
    }

    try {
        image_processor_ = std::make_unique<CLIPImageProcessor>(
                true, image_size, cv::INTER_LINEAR,
                true, image_size,
                true, 1.0 / 255.0,
                true, std::vector<double>{0.48145466, 0.4578275, 0.40821073},
                std::vector<double>{0.26862954, 0.26130258, 0.27577711}
        );
        std::cerr << "Image processor initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Image processor init failed: " << e.what() << std::endl;
        throw;
    }

    std::cerr << "Attempting to load model from: " << model_path << std::endl;
    std::ifstream file(model_path, std::ios::binary);
    if (!file.good()) {
        std::string error_msg = "File check failed: ";
        if (!std::ifstream(model_path).good()) error_msg += "File does not exist or cannot be opened";
        else if (!file.is_open()) error_msg += "Permission denied or invalid path";
        std::cerr << error_msg << std::endl;
        throw std::runtime_error(error_msg + " - Path: " + model_path);
    }

    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    file.close();
    std::cerr << "Model file size: " << file_size << " bytes" << std::endl;

    try {
        std::string abs_path = std::filesystem::absolute(model_path).string();
        std::cerr << "Loading state dict from: " << abs_path << std::endl;
        std::cerr << "Absolute path length: " << abs_path.length() << ", content: '" << abs_path << "'" << std::endl;

        std::ifstream model_file(abs_path, std::ios::binary);
        if (!model_file.is_open()) {
            std::cerr << "Failed to open model file with ifstream" << std::endl;
            throw std::runtime_error("Failed to open model file: " + abs_path);
        }

        // Load state_dict directly into an OrderedDict
        torch::OrderedDict<std::string, torch::Tensor> state_dict;
        torch::pickle_load(model_file, state_dict);
        model_file.close();

        // Load into model via underlying Module
        model_.get()->load_state_dict(state_dict);
        model_->eval();
        std::cerr << "Model loaded successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        throw std::runtime_error("Failed to load model state dict: " + std::string(e.what()));
    }

    initialize_task_prompts();
    std::cerr << "Constructor completed" << std::endl;
}

Florence2Processor::ProcessedInput Florence2Processor::preprocess(
        const std::string& text_prompt, const cv::Mat& image, const std::string& task_prompt
) {
    std::string full_prompt = get_task_prompt(task_prompt) + text_prompt;
    auto encoding = tokenizer_->encode(full_prompt, 0, false, false, "longest");
    ProcessedInput input;
    input.input_ids = encoding.ids;
    input.pixel_values = {image_processor_->preprocess(image)};
    input.attention_mask = encoding.attention_mask;
    return input;
}

Florence2Processor::ProcessedInput Florence2Processor::preprocess(
        const std::string& text_prompt, const std::vector<cv::Mat>& images, const std::string& task_prompt
) {
    std::string full_prompt = get_task_prompt(task_prompt) + text_prompt;
    auto encoding = tokenizer_->encode(full_prompt, 0, false, false, "longest");
    auto processed_images = image_processor_->preprocess(images);
    ProcessedInput input;
    input.input_ids = encoding.ids;
    input.pixel_values = processed_images;
    input.attention_mask = encoding.attention_mask;
    return input;
}

std::vector<uint32_t> Florence2Processor::generate(
        const ProcessedInput& input,
        int max_new_tokens
) {
    std::vector<int64_t> input_ids_vec(input.input_ids.begin(), input.input_ids.end());
    std::vector<int64_t> attention_mask_vec(input.attention_mask.begin(), input.attention_mask.end());
    std::vector<int64_t> decoder_input_ids_vec(input.input_ids.begin(), input.input_ids.end());
    std::vector<int64_t> decoder_attention_mask_vec(input.attention_mask.begin(), input.attention_mask.end());

    torch::Tensor pixel_values = torch::from_blob(
            input.pixel_values[0].data, {1, 3, 224, 224}, torch::kFloat
    ).to(torch::kFloat32);
    torch::Tensor input_ids = torch::tensor(input_ids_vec, torch::dtype(torch::kInt64)).unsqueeze(0);
    torch::Tensor attention_mask = torch::tensor(attention_mask_vec, torch::dtype(torch::kInt64)).unsqueeze(0);
    torch::Tensor decoder_input_ids = torch::tensor(decoder_input_ids_vec, torch::dtype(torch::kInt64)).unsqueeze(0);
    torch::Tensor decoder_attention_mask = torch::tensor(decoder_attention_mask_vec, torch::dtype(torch::kInt64)).unsqueeze(0);

    torch::NoGradGuard no_grad;
    auto logits = model_->forward(pixel_values, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask);

    std::vector<uint32_t> output_ids;
    for (int t = 0; t < max_new_tokens; t++) {
        auto next_token = logits.slice(1, -1, -1).argmax(-1).item<int64_t>();
        output_ids.push_back(static_cast<uint32_t>(next_token));
        if (next_token == 2) break;

        decoder_input_ids = torch::cat({decoder_input_ids, torch::tensor({next_token}, torch::dtype(torch::kInt64)).unsqueeze(0)}, 1);
        decoder_attention_mask = torch::cat({decoder_attention_mask, torch::ones({1, 1}, torch::dtype(torch::kInt64))}, 1);
        logits = model_->forward(pixel_values, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask);
    }

    return output_ids;
}

Florence2Processor::ProcessedOutput Florence2Processor::postprocess(
        const std::vector<uint32_t>& output_ids, const std::string& task_prompt, const cv::Mat& image
) {
    ProcessedOutput output;
    if (task_prompt == "caption") {
        output.text = tokenizer_->decode(output_ids, true);
    } else if (task_prompt == "object_detection") {
        // Simplified for now
        output.text = tokenizer_->decode(output_ids, true);
        output.bboxes = {};
    }
    return output;
}

void Florence2Processor::initialize_task_prompts() {
    task_prompts_["caption"] = "<CAPTION>";
    task_prompts_["object_detection"] = "<OD>";
}

std::string Florence2Processor::get_task_prompt(const std::string& task_prompt) {
    auto it = task_prompts_.find(task_prompt);
    return (it != task_prompts_.end()) ? it->second : "";
}
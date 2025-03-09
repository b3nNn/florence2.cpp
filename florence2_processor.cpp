#include "florence2_processor.hpp"
#include <stdexcept>

Florence2Processor::Florence2Processor(
        const std::string& tokenizer_config_path,
        const std::string& image_processor_config_path,
        CLIPImageProcessor::Size image_size
) : tokenizer_(tokenizer_config_path),
    image_processor_(
            true, image_size, cv::INTER_LINEAR,
            true, image_size, // Use same size for crop
            true, 1.0 / 255.0,
            true, {0.48145466, 0.4578275, 0.40821073},
            {0.26862954, 0.26130258, 0.27577711}
    ) {
    initialize_task_prompts();
}

void Florence2Processor::initialize_task_prompts() {
    task_prompts_ = {
            {"caption", "<CAPTION>"},
            {"detailed_caption", "<DETAILED_CAPTION>"},
            {"od", "<OD>"}, // Object detection
            {"ocr", "<OCR>"},
            {"region_caption", "<REGION_CAPTION>"},
            // Add more tasks as needed
    };
}

std::string Florence2Processor::get_task_prompt(const std::string& task_prompt) {
    auto it = task_prompts_.find(task_prompt);
    if (it != task_prompts_.end()) {
        return it->second;
    }
    return task_prompt; // If not found, assume it's already a valid prompt
}

Florence2Processor::ProcessedInput Florence2Processor::preprocess(
        const std::string& text_prompt,
        const cv::Mat& image,
        const std::string& task_prompt
) {
    std::string full_prompt = get_task_prompt(task_prompt) + text_prompt;
    auto encoded = tokenizer_.encode(full_prompt, true); // Add special tokens
    auto processed_image = image_processor_(image);

    ProcessedInput input;
    input.input_ids = encoded;
    input.pixel_values = {processed_image.pixel_values}; // Single image as batch of 1
    input.attention_mask = std::vector<uint32_t>(encoded.size(), 1); // All 1s for now
    return input;
}

Florence2Processor::ProcessedInput Florence2Processor::preprocess(
        const std::string& text_prompt,
        const std::vector<cv::Mat>& images,
        const std::string& task_prompt
) {
    std::string full_prompt = get_task_prompt(task_prompt) + text_prompt;
    auto encoded = tokenizer_.encode(full_prompt, true);
    auto processed_images = image_processor_(images);

    ProcessedInput input;
    input.input_ids = encoded;
    input.pixel_values = {processed_images.pixel_values}; // Batch of images
    input.attention_mask = std::vector<uint32_t>(encoded.size(), 1);
    return input;
}

Florence2Processor::ProcessedOutput Florence2Processor::postprocess(
        const std::vector<uint32_t>& output_ids,
        const std::string& task_prompt,
        const cv::Mat& image
) {
    ProcessedOutput output;
    std::string task = get_task_prompt(task_prompt);

    if (task == "<CAPTION>" || task == "<DETAILED_CAPTION>" || task == "<OCR>") {
        output.text = tokenizer_.decode(output_ids, true); // Skip special tokens
    } else if (task == "<OD>") {
        // Simplified object detection postprocessing (assuming output_ids encode bboxes)
        std::string decoded = tokenizer_.decode(output_ids, true);
        // Parse "<loc_x1><loc_y1><loc_x2><loc_y2>label" format (example)
        size_t pos = 0;
        while (pos < decoded.length()) {
            if (decoded.substr(pos, 4) == "<loc") {
                BoundingBox bbox;
                pos += 4; // Skip "<loc"
                bbox.bbox.push_back(std::stof(decoded.substr(pos, decoded.find(">", pos) - pos)));
                pos = decoded.find(">", pos) + 1;
                for (int i = 0; i < 3; ++i) { // y1, x2, y2
                    if (pos >= decoded.length() || decoded[pos] != '<') break;
                    pos += 4; // Skip "<loc"
                    bbox.bbox.push_back(std::stof(decoded.substr(pos, decoded.find(">", pos) - pos)));
                    pos = decoded.find(">", pos) + 1;
                }
                bbox.label = decoded.substr(pos, decoded.find("<", pos) - pos);
                bbox.score = 1.0; // Dummy score, adjust if model provides it
                output.bboxes.push_back(bbox);
                pos = decoded.find("<", pos);
                if (pos == std::string::npos) break;
            } else {
                pos++;
            }
        }
    } else {
        throw std::runtime_error("Unsupported task prompt: " + task);
    }

    return output;
}
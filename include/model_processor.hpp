#ifndef MODEL_PROCESSOR_HPP
#define MODEL_PROCESSOR_HPP

#include <string>
#include <vector>
#include <utility>
#include "image.hpp"
#include "tokenizer.hpp"
#include "processor_output.hpp"
class ModelProcessor {
public:
    ModelProcessor() = default;

    // Static method to initialize the ModelProcessor from a pretrained model
    static ModelProcessor from_pretrained(const std::string& model_name, bool trust_remote_code);

    // Method to process text and image inputs
    ProcessorOutput *process(const std::string& text, const Image& image);

    std::vector<uint32_t> generate(const std::vector<uint32_t>& input_ids, const std::vector<uint32_t>& pixel_values, 
                              int max_new_tokens, int num_beams, bool do_sample);

    // Method to decode generated IDs into text
    std::string batch_decode(const std::vector<uint32_t>& generated_ids, bool skip_special_tokens);

    // Method to post-process the generated text
    std::string post_process_generation(const std::string& generated_text, const std::string& task, 
                                       const std::pair<int, int>& image_size);
};

#endif // MODEL_PROCESSOR_HPP
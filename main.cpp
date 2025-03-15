#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include "image.hpp"
#include "model_processor.hpp"

int main() {
    // Initialize the ModelProcessor
    ModelProcessor model_processor = ModelProcessor::from_pretrained("microsoft/Florence-2-large", true);

    // Define the prompt
    std::string prompt = "<OD>";

    // Download and open the image
    std::string url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true";
    Image image = Image::open(url);

    // Process the inputs
    auto processor_output = model_processor.process(prompt, image);

    // Generate output
    auto generated_ids = model_processor.generate(processor_output->input_ids, processor_output->pixel_values, 4096, 3, false);

    delete processor_output;

    // Decode the generated IDs
    std::string generated_text = model_processor.batch_decode(generated_ids, false);

    // Post-process the generated text
    std::string parsed_answer = model_processor.post_process_generation(generated_text, "<OD>", 
                                                                       std::make_pair(image.width(), image.height()));

    // Print the result
    std::cout << parsed_answer << std::endl;

    return 0;
}
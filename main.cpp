#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include "image.hpp"
#include "model_processor.hpp"

int main() {
    // Determine device and data type
    std::string device = "cpu"; // Placeholder: In real code, check for CUDA availability
    std::string torch_dtype = "float32"; // Placeholder: In real code, check for CUDA and use float16 if available

    // Initialize the ModelProcessor
    ModelProcessor model_processor = ModelProcessor::from_pretrained("microsoft/Florence-2-large", true);
    model_processor.to(device, torch_dtype);

    // Define the prompt
    std::string prompt = "<OD>";

    // Download and open the image
    std::string url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true";
    Image image = Image::open(url);

    // Process the inputs
    auto inputs = model_processor.process(prompt, image, "pt");

    // Placeholder: Simulate extracting input_ids and pixel_values from inputs
    std::vector<int> input_ids; // In real code, extract from inputs
    std::vector<int> pixel_values; // In real code, extract from inputs

    // Generate output
    auto generated_ids = model_processor.generate(input_ids, pixel_values, 4096, 3, false);

    // Decode the generated IDs
    std::string generated_text = model_processor.batch_decode(generated_ids, false);

    // Post-process the generated text
    std::string parsed_answer = model_processor.post_process_generation(generated_text, "<OD>", 
                                                                       std::make_pair(image.width(), image.height()));

    // Print the result
    std::cout << parsed_answer << std::endl;

    return 0;
}
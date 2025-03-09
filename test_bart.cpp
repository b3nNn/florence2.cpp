#include <iostream>
#include "hf_tokenizers.hpp"
#include "tokenizers.h"

int main() {
    try {
        // Replace with the path to your tokenizer.json file
        Tokenizer tokenizer("tokenizer.json");

        // Encode text
        std::string text = "Hello, world! This is my final test to ensure that the tokenizer is working propertly. Thanks to grok for the help.";
        std::vector<uint32_t> ids = tokenizer.encode(text).ids;
        std::cout << "Token IDs: ";
        for (uint32_t id : ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;

        // Decode back to text
        std::string decoded = tokenizer.decode(ids, false);
        std::cout << "Decoded text: " << decoded << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
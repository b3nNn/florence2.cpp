#include "florence2_processor.hpp"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <random>
#include <torch/torch.h>

int main() {
    try {
        Florence2Processor::Florence2Processor processor(
            "florence2.onnx",
            "tokenizer.json",
            "preprocessor_config.json"
        );

        cv::Mat image = cv::imread("puma.png");
        if (image.empty()) {
            throw std::runtime_error("Failed to load image 'puma.png'");
        }

        std::string text = "<CAPTION>";
        std::vector<float> logits = processor.process(text, image);

        std::cout << "Inference output size: " << logits.size() << std::endl;
        std::cout << "First few logits: ";
        for (size_t i = 0; i < std::min<size_t>(5, logits.size()); ++i) {
            std::cout << logits[i] << " ";
        }
        std::cout << std::endl;

        const int64_t batch_size = 1;
        const int64_t seq_length = 256;
        const int64_t vocab_size = 50265;
        size_t expected_size = batch_size * seq_length * vocab_size;
        if (logits.size() < expected_size) {
            throw std::runtime_error("Logits size too small");
        }

        torch::Tensor logits_tensor = torch::from_blob(logits.data(), {batch_size, seq_length, vocab_size});
        std::cout << "Logits tensor shape: " << logits_tensor.sizes() << std::endl;

        // Take first 5 tokens as caption, force EOS
        std::vector<uint32_t> token_ids;
        const int max_caption_len = 5;  // Typical caption length
        for (int64_t i = 0; i < max_caption_len; ++i) {
            torch::Tensor step_logits = logits_tensor[0][i].to(torch::kCPU);
            auto [max_val, max_idx] = step_logits.max(0);
            uint32_t token_id = static_cast<uint32_t>(max_idx.item<int64_t>());
            float eos_logit = step_logits[2].item<float>();
            std::cout << "Step " << i << " - Token: " << token_id
                      << ", Max logit: " << max_val.item<float>()
                      << ", EOS logit: " << eos_logit << std::endl;
            token_ids.push_back(token_id);
        }
        token_ids.push_back(2);  // Append EOS manually

        std::cout << "Token IDs size: " << token_ids.size() << std::endl;
        std::cout << "Token IDs: ";
        for (size_t i = 0; i < token_ids.size(); ++i) {
            std::cout << token_ids[i] << " ";
        }
        std::cout << std::endl;

        std::string caption = processor.decode(token_ids, true);
        std::cout << "Generated caption: " << caption << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
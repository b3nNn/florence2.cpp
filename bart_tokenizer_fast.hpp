#ifndef BART_TOKENIZER_FAST_HPP
#define BART_TOKENIZER_FAST_HPP

#include "hf_tokenizers.hpp"
#include <string>
#include <vector>
#include <map>
#include <set>

class BartTokenizerFast {
public:
    // Constructor: Initialize with the path to the tokenizer configuration directory
    BartTokenizerFast(const std::string& config_path);

    // Encode text into token IDs, with optional BART-specific processing
    std::vector<uint32_t> encode(
            const std::string& text,
            bool add_special_tokens = true,
            size_t max_length = 0,
            bool truncation = false,
            bool padding = false,
            const std::string& padding_strategy = "longest"
    );

    // Decode token IDs back to text, with optional skipping of special tokens
    std::string decode(const std::vector<uint32_t>& ids, bool skip_special_tokens = true);

    // Get the vocabulary size
    size_t get_vocab_size() const;

    // Add a special token (if needed)
    void add_special_token(const std::string& token, uint32_t id);

    // Get the ID of a special token
    uint32_t get_special_token_id(const std::string& token) const;

    // Get all special tokens
    std::vector<std::pair<std::string, uint32_t>> get_special_tokens() const;

private:
    // Pre-process text (e.g., handle whitespace, normalization)
    std::string preprocess(const std::string& text) const;

    // Post-process token IDs (e.g., add special tokens like <s> and </s>)
    std::vector<uint32_t> postprocess(const std::vector<uint32_t>& ids, bool add_special_tokens) const;

    // Check if a token ID is a special token
    bool is_special_token(uint32_t id) const;

    // Underlying tokenizer from your wrapper
    Tokenizer tokenizer_;

    // Special tokens and their IDs (BART-specific)
    std::map<std::string, uint32_t> special_tokens_;
    std::set<uint32_t> special_token_ids_;

    // BART-specific special token IDs
    uint32_t bos_token_id_;  // <s>
    uint32_t eos_token_id_;  // </s>
    uint32_t pad_token_id_;  // <pad>
    uint32_t unk_token_id_;  // <unk>
    uint32_t mask_token_id_; // <mask>
};

#endif // BART_TOKENIZER_FAST_HPP
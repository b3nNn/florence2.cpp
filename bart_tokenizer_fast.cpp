#include "bart_tokenizer_fast.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include "tokenizers.h"

BartTokenizerFast::BartTokenizerFast(const std::string& config_path)
        : tokenizer_(config_path) {
    // Initialize BART-specific special tokens (these are the defaults for facebook/bart-base)
    // These IDs are typically defined in the tokenizer configuration, but we hardcode them for clarity
    // In a real implementation, you might want to load these from a config file (e.g., tokenizer.json)
    bos_token_id_ = 0;   // <s>
    eos_token_id_ = 2;   // </s>
    pad_token_id_ = 1;   // <pad>
    unk_token_id_ = 3;   // <unk>
    mask_token_id_ = 50264; // <mask>

    // Add special tokens to the map
    add_special_token("<s>", bos_token_id_);
    add_special_token("</s>", eos_token_id_);
    add_special_token("<pad>", pad_token_id_);
    add_special_token("<unk>", unk_token_id_);
    add_special_token("<mask>", mask_token_id_);
}

std::vector<uint32_t> BartTokenizerFast::encode(
        const std::string& text,
        bool add_special_tokens,
        size_t max_length,
        bool truncation,
        bool padding,
        const std::string& padding_strategy
) {
    return tokenizer_.encode(text, max_length, truncation, padding, padding_strategy).ids;
}

std::string BartTokenizerFast::decode(const std::vector<uint32_t>& ids, bool skip_special_tokens) {
    if (!skip_special_tokens) {
        // If not skipping special tokens, just use the underlying tokenizer's decode
        return tokenizer_.decode(ids, false);
    }

    // Filter out special tokens
    std::vector<uint32_t> filtered_ids;
    for (uint32_t id : ids) {
        if (!is_special_token(id)) {
            filtered_ids.push_back(id);
        }
    }

    // Decode the filtered IDs
    return tokenizer_.decode(filtered_ids, false);
}

size_t BartTokenizerFast::get_vocab_size() const {
    // Note: Your wrapper doesn't expose vocab size directly, so this is a placeholder
    // In a real implementation, you might need to extend your wrapper to expose this
    throw std::runtime_error("get_vocab_size not implemented in wrapper");
    return 0;
}

void BartTokenizerFast::add_special_token(const std::string& token, uint32_t id) {
    special_tokens_[token] = id;
    special_token_ids_.insert(id);
}

uint32_t BartTokenizerFast::get_special_token_id(const std::string& token) const {
    auto it = special_tokens_.find(token);
    if (it == special_tokens_.end()) {
        throw std::runtime_error("Special token not found: " + token);
    }
    return it->second;
}

std::string BartTokenizerFast::preprocess(const std::string& text) const {
    // Basic pre-processing: handle whitespace, etc.
    // Note: The Hugging Face tokenizers library handles most pre-processing internally,
    // so this is minimal. You might need to add more depending on your use case.
    std::stringstream ss;
    bool first = true;
    for (char c : text) {
        if (std::isspace(c)) {
            if (!first) ss << " ";
            first = true;
        } else {
            ss << c;
            first = false;
        }
    }
    return ss.str();
}

std::vector<uint32_t> BartTokenizerFast::postprocess(const std::vector<uint32_t>& ids, bool add_special_tokens) const {
    if (!add_special_tokens) {
        return ids;
    }

    // Add BART-specific special tokens: <s> at the start, </s> at the end
    std::vector<uint32_t> processed_ids;
    processed_ids.push_back(bos_token_id_); // Add <s>
    processed_ids.insert(processed_ids.end(), ids.begin(), ids.end()); // Add original IDs
    processed_ids.push_back(eos_token_id_); // Add </s>
    return processed_ids;
}

bool BartTokenizerFast::is_special_token(uint32_t id) const {
    return special_token_ids_.find(id) != special_token_ids_.end();
}

//std::vector <std::string, uint32_t>

std::vector<std::pair<std::string, uint32_t>> BartTokenizerFast::get_special_tokens() const {
    char** tokens;
    uint32_t* ids;
    size_t count;
    if (!tokenizer_get_special_tokens(tokenizer_.get_raw_tokenizer(), &tokens, &ids, &count)) {
        return {};
    }
    std::vector<std::pair<std::string, uint32_t>> result;
    for (size_t i = 0; i < count; i++) {
        result.emplace_back(tokens[i], ids[i]);
    }
    free_special_tokens(tokens, ids, count);
    return result;
}

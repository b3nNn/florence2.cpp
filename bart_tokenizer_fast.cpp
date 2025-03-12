#include "bart_tokenizer_fast.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include "tokenizers.h"

BartTokenizerFast::BartTokenizerFast(const std::string& config_path)
        : tokenizer_(config_path) {
    bos_token_id_ = 0;   // <s>
    eos_token_id_ = 2;   // </s>
    pad_token_id_ = 1;   // <pad>
    unk_token_id_ = 3;   // <unk>
    mask_token_id_ = 50264; // <mask>

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
        return tokenizer_.decode(ids, false);
    }
    std::vector<uint32_t> filtered_ids;
    for (uint32_t id : ids) {
        if (!is_special_token(id)) {
            filtered_ids.push_back(id);
        }
    }
    return tokenizer_.decode(filtered_ids, false);
}

void BartTokenizerFast::add_special_tokens(const std::vector<std::string>& tokens) {
    // Convert std::vector<std::string> to char* array
    std::vector<const char*> c_tokens;
    c_tokens.reserve(tokens.size());
    for (const auto& token : tokens) {
        c_tokens.push_back(token.c_str());
    }

    // Add tokens to underlying tokenizer
    tokenizer_add_special_tokens(tokenizer_.get_raw_tokenizer(), c_tokens.data(), c_tokens.size());

    // Update special tokens map
    size_t vocab_size = get_vocab_size();
    for (size_t i = 0; i < tokens.size(); ++i) {
        uint32_t id = vocab_size - tokens.size() + i;  // New IDs at end
        special_tokens_[tokens[i]] = id;
        special_token_ids_.insert(id);
    }
}

size_t BartTokenizerFast::get_vocab_size() const {
    return tokenizer_get_vocab_size(tokenizer_.get_raw_tokenizer());
}

void BartTokenizerFast::add_special_token(const std::string& token, uint32_t id) {
    tokenizer_.add_special_token(token);
}

uint32_t BartTokenizerFast::get_special_token_id(const std::string& token) const {
    auto it = special_tokens_.find(token);
    if (it == special_tokens_.end()) {
        throw std::runtime_error("Special token not found: " + token);
    }
    return it->second;
}

std::string BartTokenizerFast::preprocess(const std::string& text) const {
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
    std::vector<uint32_t> processed_ids;
    processed_ids.push_back(bos_token_id_);
    processed_ids.insert(processed_ids.end(), ids.begin(), ids.end());
    processed_ids.push_back(eos_token_id_);
    return processed_ids;
}

bool BartTokenizerFast::is_special_token(uint32_t id) const {
    return special_token_ids_.find(id) != special_token_ids_.end();
}

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
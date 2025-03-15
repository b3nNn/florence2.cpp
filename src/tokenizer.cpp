#include "tokenizer.hpp"
#include <stdexcept>
#include <cstring>

Tokenizer::Tokenizer(const std::string& file_path) {
    tokenizer_ = tokenizer_from_file(file_path.c_str());
    if (!tokenizer_) {
        throw std::runtime_error("Failed to load tokenizer from file: " + file_path);
    }
    char** tokens = nullptr;
    uint32_t* ids = nullptr;
    size_t count = 0;
    if (tokenizer_get_special_tokens(tokenizer_, &tokens, &ids, &count)) {
        for (size_t i = 0; i < count; ++i) {
            special_tokens_[std::string(tokens[i])] = ids[i];
            special_token_ids_[ids[i]] = true;
        }
        free_special_tokens(tokens, ids, count);
    }
}

Tokenizer::~Tokenizer() {
    if (tokenizer_) {
        tokenizer_free(tokenizer_);
    }
}

Tokenizer::Encoding Tokenizer::encode(
        const std::string& text,
        size_t max_length,
        bool truncation,
        bool padding,
        const std::string& padding_strategy
) {
    uint32_t* ids = nullptr;
    size_t ids_len = 0;
    uint32_t* attention_mask = nullptr;
    size_t mask_len = 0;

    if (!tokenizer_encode(
            tokenizer_, text.c_str(), &ids, &ids_len, &attention_mask, &mask_len,
            max_length, truncation, padding, padding_strategy.c_str()
    )) {
        throw std::runtime_error("Failed to encode text");
    }

    Encoding result;
    result.ids = std::vector<uint32_t>(ids, ids + ids_len);
    result.attention_mask = std::vector<uint32_t>(attention_mask, attention_mask + mask_len);

    free_token_ids(ids);
    free_token_ids(attention_mask);
    return result;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids, bool skip_special_tokens) {
    char* result = tokenizer_decode(tokenizer_, ids.data(), ids.size(), skip_special_tokens);
    if (!result) {
        throw std::runtime_error("Failed to decode token IDs");
    }
    std::string decoded = result;
    free_c_string(result);
    return decoded;
}

size_t Tokenizer::get_vocab_size() const {
    return tokenizer_get_vocab_size(tokenizer_);
}

std::optional<uint32_t> Tokenizer::get_special_token_id(const std::string& token) const {
    uint32_t id;
    if (tokenizer_get_special_token_id(tokenizer_, token.c_str(), &id)) {
        return id;
    }
    return std::nullopt;
}

uint32_t Tokenizer::add_special_token(const std::string& token) {
    uint32_t id = tokenizer_add_special_token(tokenizer_, token.c_str());
    if (id != UINT32_MAX) {
        special_tokens_[token] = id;
        special_token_ids_[id] = true;
    }
    return id;
}

void Tokenizer::add_special_tokens(const std::vector<std::string>& tokens) {
    std::vector<const char*> c_tokens;
    c_tokens.reserve(tokens.size());
    for (const auto& token : tokens) {
        c_tokens.push_back(token.c_str());
    }
    size_t num_added = tokenizer_add_special_tokens(tokenizer_, c_tokens.data(), c_tokens.size());
    if (num_added != tokens.size()) {
        throw std::runtime_error("Failed to add all special tokens");
    }
    // Update special tokens map
    for (const auto& token : tokens) {
        if (auto id = get_special_token_id(token)) {
            special_tokens_[token] = *id;
            special_token_ids_[*id] = true;
        }
    }
}

bool Tokenizer::is_special_token(uint32_t id) const {
    auto it = special_token_ids_.find(id);
    if (it != special_token_ids_.end()) {
        return true;
    }
    return tokenizer_is_special_token(tokenizer_, id);
}
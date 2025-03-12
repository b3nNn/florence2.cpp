#ifndef BART_TOKENIZER_FAST_HPP
#define BART_TOKENIZER_FAST_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "hf_tokenizers.hpp"

class BartTokenizerFast {
public:
    BartTokenizerFast(const std::string& config_path);
    std::vector<uint32_t> encode(
            const std::string& text,
            bool add_special_tokens,
            size_t max_length,
            bool truncation,
            bool padding,
            const std::string& padding_strategy
    );
    std::string decode(const std::vector<uint32_t>& ids, bool skip_special_tokens);
    void add_special_tokens(const std::vector<std::string>& tokens);
    size_t get_vocab_size() const;
    uint32_t get_special_token_id(const std::string& token) const;
    std::string preprocess(const std::string& text) const;
    std::vector<uint32_t> postprocess(const std::vector<uint32_t>& ids, bool add_special_tokens) const;
    bool is_special_token(uint32_t id) const;
    std::vector<std::pair<std::string, uint32_t>> get_special_tokens() const;

private:
    Tokenizer tokenizer_;
    std::unordered_map<std::string, uint32_t> special_tokens_;
    std::unordered_set<uint32_t> special_token_ids_;
    uint32_t bos_token_id_;
    uint32_t eos_token_id_;
    uint32_t pad_token_id_;
    uint32_t unk_token_id_;
    uint32_t mask_token_id_;
    void add_special_token(const std::string& token, uint32_t id);
};

#endif // BART_TOKENIZER_FAST_HPP
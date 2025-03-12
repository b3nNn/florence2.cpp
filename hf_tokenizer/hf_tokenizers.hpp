#ifndef HF_TOKENIZERS_HPP
#define HF_TOKENIZERS_HPP

#include <string>
#include <vector>
#include <map>
#include <optional>

struct CTokenizer;

class Tokenizer {
public:
    Tokenizer(const std::string& file_path);
    ~Tokenizer();

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    struct Encoding {
        std::vector<uint32_t> ids;
        std::vector<uint32_t> attention_mask;
    };
    Encoding encode(
            const std::string& text,
            size_t max_length = 0,
            bool truncation = false,
            bool padding = false,
            const std::string& padding_strategy = "longest"
    );

    std::string decode(const std::vector<uint32_t>& ids, bool skip_special_tokens = true);
    size_t get_vocab_size() const;
    std::optional<uint32_t> get_special_token_id(const std::string& token) const;
    uint32_t add_special_token(const std::string& token);
    void add_special_tokens(const std::vector<std::string>& tokens);  // New batch method
    bool is_special_token(uint32_t id) const;
    CTokenizer* get_raw_tokenizer() const { return tokenizer_; }

private:
    CTokenizer* tokenizer_ = nullptr;
    std::map<std::string, uint32_t> special_tokens_;
    std::map<uint32_t, bool> special_token_ids_;
};

#endif // HF_TOKENIZERS_HPP
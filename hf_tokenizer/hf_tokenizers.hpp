#ifndef HF_TOKENIZERS_HPP
#define HF_TOKENIZERS_HPP

#include <string>
#include <vector>
#include <map>
#include <optional>

// Forward declaration of C bindings (assumed to be provided by the tokenizers library)
struct CTokenizer;
struct CEncoding;

class Tokenizer {
public:
    Tokenizer(const std::string& file_path);
    ~Tokenizer();

    // Disable copying to prevent double-free
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Encode text into token IDs, with optional truncation, padding, and attention masks
    struct Encoding {
        std::vector<uint32_t> ids;              // Token IDs
        std::vector<uint32_t> attention_mask;   // Attention mask (1 for real tokens, 0 for padding)
    };
    Encoding encode(
            const std::string& text,
            size_t max_length = 0,                  // 0 means no truncation
            bool truncation = false,                // Enable truncation
            bool padding = false,                   // Enable padding
            const std::string& padding_strategy = "longest" // "longest", "max_length", "do_not_pad"
    );

    // Decode token IDs back to text
    std::string decode(const std::vector<uint32_t>& ids, bool skip_special_tokens = true);

    // Get vocabulary size
    size_t get_vocab_size() const;

    // Get special token IDs
    std::optional<uint32_t> get_special_token_id(const std::string& token) const;

    // Add a special token (returns the assigned ID)
    uint32_t add_special_token(const std::string& token);

    // Check if a token ID is a special token
    bool is_special_token(uint32_t id) const;

    // Get the tokenizer
    CTokenizer* get_raw_tokenizer() const { return tokenizer_; }

private:
    CTokenizer* tokenizer_ = nullptr;

    // Cache of special token IDs (loaded during initialization)
    std::map<std::string, uint32_t> special_tokens_;
    std::map<uint32_t, bool> special_token_ids_; // For fast lookup
};

#endif // HF_TOKENIZERS_HPP
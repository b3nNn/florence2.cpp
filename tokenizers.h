#ifndef TOKENIZERS_H
#define TOKENIZERS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CTokenizer CTokenizer;

// Load a tokenizer from a JSON file
CTokenizer* tokenizer_from_file(const char* file_path);

// Free the tokenizer
void tokenizer_free(CTokenizer* tokenizer);

// Encode text into token IDs (updated for attention mask)
bool tokenizer_encode(
        CTokenizer* tokenizer,
        const char* text,
        uint32_t** ids_out,
        size_t* len_out,
        uint32_t** attention_mask_out,
        size_t* mask_len_out,
        size_t max_length,
        bool truncation,
        bool padding,
        const char* padding_strategy
);

// Free the token IDs array
void free_token_ids(uint32_t* ids);

// Decode token IDs back to text
char* tokenizer_decode(CTokenizer* tokenizer, const uint32_t* ids, size_t len, bool skip_special_tokens);

// Free a C string allocated by Rust
void free_c_string(char* s);

// Get the vocabulary size
size_t tokenizer_get_vocab_size(CTokenizer* tokenizer);

// Get a special token ID
bool tokenizer_get_special_token_id(CTokenizer* tokenizer, const char* token, uint32_t* id_out);

// Add a single special token
uint32_t tokenizer_add_special_token(CTokenizer* tokenizer, const char* token);

// Add multiple special tokens
size_t tokenizer_add_special_tokens(CTokenizer* tokenizer, const char** tokens, size_t count);

// Check if a token ID is a special token
bool tokenizer_is_special_token(CTokenizer* tokenizer, uint32_t id);

// Get all special tokens
bool tokenizer_get_special_tokens(
        CTokenizer* tokenizer,
        char*** tokens_out,
        uint32_t** ids_out,
        size_t* count_out
);

// Free the special tokens arrays
void free_special_tokens(char** tokens, uint32_t* ids, size_t count);

#ifdef __cplusplus
}
#endif

#endif  /* TOKENIZERS_H */

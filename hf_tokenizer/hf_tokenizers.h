#ifndef HF_TOKENIZERS_H
#define HF_TOKENIZERS_H

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

// Encode text into token IDs
bool tokenizer_encode(CTokenizer* tokenizer, const char* text, uint32_t** ids_out, size_t* len_out);

// Free the token IDs array
void free_token_ids(uint32_t* ids);

// Decode token IDs back to text
char* tokenizer_decode(CTokenizer* tokenizer, const uint32_t* ids, size_t len, bool skip_special_tokens);

// Free a C string allocated by Rust
void free_c_string(char* s);

#ifdef __cplusplus
}
#endif

#endif // HF_TOKENIZERS_H
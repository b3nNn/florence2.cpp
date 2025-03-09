#ifndef TOKENIZERS_H
#define TOKENIZERS_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CTokenizer {
  uint8_t _private[0];
} CTokenizer;

void free_c_string(char *s);

void free_token_ids(uint32_t *ids);

struct CTokenizer *tokenizer_from_file(const char *file_path);

void tokenizer_free(struct CTokenizer *tokenizer);

bool tokenizer_encode(struct CTokenizer *tokenizer,
                      const char *text,
                      uint32_t **ids_out,
                      uintptr_t *ids_len_out,
                      uint32_t **attention_mask_out,
                      uintptr_t *mask_len_out,
                      uintptr_t max_length,
                      bool truncation,
                      bool padding,
                      const char *padding_strategy);

char *tokenizer_decode(struct CTokenizer *tokenizer,
                       const uint32_t *ids,
                       uintptr_t len,
                       bool skip_special_tokens);

uintptr_t tokenizer_get_vocab_size(struct CTokenizer *tokenizer);

bool tokenizer_get_special_token_id(struct CTokenizer *tokenizer,
                                    const char *token,
                                    uint32_t *id_out);

uint32_t tokenizer_add_special_token(struct CTokenizer *tokenizer, const char *token);

bool tokenizer_is_special_token(struct CTokenizer *tokenizer, uint32_t id);

bool tokenizer_get_special_tokens(struct CTokenizer *tokenizer,
                                  char ***tokens_out,
                                  uint32_t **ids_out,
                                  uintptr_t *count_out);

void free_special_tokens(char **tokens, uint32_t *ids, uintptr_t count);

#ifdef __cplusplus
}
#endif

#endif  /* TOKENIZERS_H */

#ifndef POSITIONAL_EMBEDDING_COSINE_1D_H
#define POSITIONAL_EMBEDDING_COSINE_1D_H

#include <torch/torch.h>

class PositionalEmbeddingCosine1D : public torch::nn::Module {
public:
    PositionalEmbeddingCosine1D(int embed_dim = 512, int max_seq_len = 1024);

    torch::Tensor forward(torch::Tensor seq_embeds);

private:
    int embed_dim_;
    int max_seq_len_;
    torch::Tensor pos_idx_to_embed_;
};

#endif // POSITIONAL_EMBEDDING_COSINE_1D_H
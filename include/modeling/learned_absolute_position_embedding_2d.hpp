#ifndef LEARNED_ABSOLUTE_POSITION_EMBEDDING_2D_H
#define LEARNED_ABSOLUTE_POSITION_EMBEDDING_2D_H

#include <torch/torch.h>

class LearnedAbsolutePositionEmbedding2D : public torch::nn::Module {
public:
    // Constructor
    LearnedAbsolutePositionEmbedding2D(int64_t embedding_dim, int64_t num_pos);

    // Forward method
    torch::Tensor forward(torch::Tensor pixel_values);

private:
    torch::nn::Embedding row_embeddings{nullptr};
    torch::nn::Embedding column_embeddings{nullptr};
};

#endif // LEARNED_ABSOLUTE_POSITION_EMBEDDING_2D_H
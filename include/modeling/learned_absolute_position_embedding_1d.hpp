#ifndef LEARNED_ABSOLUTE_POSITION_EMBEDDING_H
#define LEARNED_ABSOLUTE_POSITION_EMBEDDING_H

#include <torch/torch.h>

// Class definition for LearnedAbsolutePositionEmbedding1D
class LearnedAbsolutePositionEmbedding1D : public torch::nn::Module {
public:
    // Constructor
    LearnedAbsolutePositionEmbedding1D(int embedding_dim = 512, int num_pos = 1024);

    // Forward method
    torch::Tensor forward(torch::Tensor seq_embeds);

private:
    // Embedding layer for positional encodings
    torch::nn::Embedding embeddings{nullptr};
    // Maximum number of positions
    int num_pos;
};

#endif // LEARNED_ABSOLUTE_POSITION_EMBEDDING_H
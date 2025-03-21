#include "learned_absolute_position_embedding.h"

// Constructor implementation
LearnedAbsolutePositionEmbedding1D::LearnedAbsolutePositionEmbedding1D(int embedding_dim, int num_pos)
    : num_pos(num_pos) {
    // Initialize the embedding layer
    embeddings = register_module("embeddings", torch::nn::Embedding(num_pos, embedding_dim));
}

// Forward method implementation
torch::Tensor LearnedAbsolutePositionEmbedding1D::forward(torch::Tensor seq_embeds) {
    // Check input dimensions (must be 2D or 3D)
    int shape_len = seq_embeds.dim();
    TORCH_CHECK(2 <= shape_len && shape_len <= 3, "Input must be 2D or 3D");

    // Get sequence length from the second-last dimension
    int64_t len_seq = seq_embeds.size(-2);
    TORCH_CHECK(len_seq <= num_pos, "Sequence length exceeds maximum position");

    // Create position indices tensor on the same device as input
    auto positions = torch::arange(len_seq, torch::TensorOptions()
                                        .dtype(torch::kLong)
                                        .device(seq_embeds.device()));

    // Compute positional embeddings
    torch::Tensor pos_embeds = embeddings->forward(positions);

    // If input is 3D (batched), add batch dimension to positional embeddings
    if (shape_len == 3) {
        pos_embeds = pos_embeds.view({1, pos_embeds.size(0), pos_embeds.size(1)});
    }

    return pos_embeds;
}
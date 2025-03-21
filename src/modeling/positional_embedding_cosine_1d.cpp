#include <cmath>
#include "positional_embedding_cosine_1d.hpp"

PositionalEmbeddingCosine1D::PositionalEmbeddingCosine1D(int embed_dim, int max_seq_len)
    : embed_dim_(embed_dim), max_seq_len_(max_seq_len) {
    // Compute positional embeddings
    double factor = std::log(10000.0);
    torch::Tensor denominator = torch::exp(
        -factor * torch::arange(0, embed_dim_, 2) / static_cast<double>(embed_dim_));

    torch::Tensor frequencies = torch::arange(0, max_seq_len_).view({max_seq_len_, 1}) * denominator;

    pos_idx_to_embed_ = torch::zeros({max_seq_len_, embed_dim_});

    // Populate uneven entries
    auto sin_part = pos_idx_to_embed_.slice(1, 0, embed_dim_, 2);
    sin_part.copy_(torch::sin(frequencies));

    auto cos_part = pos_idx_to_embed_.slice(1, 1, embed_dim_, 2);
    cos_part.copy_(torch::cos(frequencies));

    // Register the buffer
    register_buffer("pos_idx_to_embed", pos_idx_to_embed_);
}

torch::Tensor PositionalEmbeddingCosine1D::forward(torch::Tensor seq_embeds) {
    int shape_len = seq_embeds.dim();
    TORCH_CHECK(2 <= shape_len && shape_len <= 3, "Input must be 2D or 3D");
    int len_seq = seq_embeds.size(shape_len - 2);
    TORCH_CHECK(len_seq <= max_seq_len_, "Sequence length exceeds max_seq_len");

    auto pos_embeds = pos_idx_to_embed_.slice(0, 0, len_seq);
    if (shape_len == 3) {
        pos_embeds = pos_embeds.unsqueeze(0);
    }
    return pos_embeds;
}
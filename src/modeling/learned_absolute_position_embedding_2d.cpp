#include "learned_absolute_position_embedding_2d.hpp"

LearnedAbsolutePositionEmbedding2D::LearnedAbsolutePositionEmbedding2D(int64_t embedding_dim, int64_t num_pos) {
    row_embeddings = register_module("row_embeddings", torch::nn::Embedding(num_pos, embedding_dim / 2));
    column_embeddings = register_module("column_embeddings", torch::nn::Embedding(num_pos, embedding_dim - (embedding_dim / 2)));
}

torch::Tensor LearnedAbsolutePositionEmbedding2D::forward(torch::Tensor pixel_values) {
    TORCH_CHECK(pixel_values.dim() == 4, "pixel_values must be a 4D tensor");

    int64_t batch_size = pixel_values.size(0);
    int64_t height = pixel_values.size(1);
    int64_t width = pixel_values.size(2);

    auto options = torch::TensorOptions().dtype(torch::kLong).device(pixel_values.device());
    auto width_values = torch::arange(0, width, options);
    auto height_values = torch::arange(0, height, options);

    auto x_emb = column_embeddings(width_values);
    auto y_emb = row_embeddings(height_values);

    auto x_emb_expanded = x_emb.unsqueeze(0).repeat({height, 1, 1});
    auto y_emb_expanded = y_emb.unsqueeze(1).repeat({1, width, 1});

    auto pos = torch::cat({x_emb_expanded, y_emb_expanded}, 2);
    pos = pos.permute({2, 0, 1});
    pos = pos.unsqueeze(0);
    pos = pos.repeat({batch_size, 1, 1, 1});
    pos = pos.permute({0, 2, 3, 1});

    return pos;
}
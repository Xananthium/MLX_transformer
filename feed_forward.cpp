#include "feed_forward.h"

#include <mlx/ops.h>
#include <mlx/nn/layers.h>

namespace mlx_transformer {

FeedForward::FeedForward(int64_t hidden_size, int64_t intermediate_size, float dropout_prob)
    : hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      dropout_prob_(dropout_prob) {
    
    // Initialize feed-forward weights
    gate_weight_ = mlx::core::zeros({hidden_size_, intermediate_size_}, mlx::core::float32);
    up_weight_ = mlx::core::zeros({hidden_size_, intermediate_size_}, mlx::core::float32);
    down_weight_ = mlx::core::zeros({intermediate_size_, hidden_size_}, mlx::core::float32);
}

void FeedForward::loadWeights(ModelLoader& loader, const std::string& prefix) {
    gate_weight_ = loader.loadWeight(prefix + ".gate_proj.weight");
    up_weight_ = loader.loadWeight(prefix + ".up_proj.weight");
    down_weight_ = loader.loadWeight(prefix + ".down_proj.weight");
}

mlx::core::array FeedForward::forward(const mlx::core::array& hidden_states) {
    // SwiGLU activation as used in many modern transformer models
    auto gate = mlx::core::matmul(hidden_states, gate_weight_);
    gate = mlx::core::gelu(gate);
    
    auto up = mlx::core::matmul(hidden_states, up_weight_);
    auto intermediate = mlx::core::multiply(gate, up);
    
    // Project back to hidden dimension
    auto output = mlx::core::matmul(intermediate, down_weight_);
    
    // Apply dropout if needed
    if (dropout_prob_ > 0.0) {
        output = mlx::nn::dropout(output, dropout_prob_);
    }
    
    return output;
}

} // namespace mlx_transformer
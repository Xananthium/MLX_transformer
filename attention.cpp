#include "attention.h"

#include <mlx/ops.h>
#include <mlx/nn/attention.h>
#include <cmath>

namespace mlx_transformer {

AttentionImplementation::AttentionImplementation(
    int64_t hidden_size,
    int64_t num_heads,
    float dropout_prob)
    : hidden_size_(hidden_size),
      num_heads_(num_heads),
      head_dim_(hidden_size / num_heads),
      dropout_prob_(dropout_prob),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    
    // Initialize query, key, value, and output projection weights
    // These would normally be loaded from the model
    query_weight_ = mlx::core::zeros({hidden_size_, hidden_size_}, mlx::core::float32);
    key_weight_ = mlx::core::zeros({hidden_size_, hidden_size_}, mlx::core::float32);
    value_weight_ = mlx::core::zeros({hidden_size_, hidden_size_}, mlx::core::float32);
    output_weight_ = mlx::core::zeros({hidden_size_, hidden_size_}, mlx::core::float32);
}

void AttentionImplementation::loadWeights(ModelLoader& loader, const std::string& prefix) {
    query_weight_ = loader.loadWeight(prefix + ".wq.weight");
    key_weight_ = loader.loadWeight(prefix + ".wk.weight");
    value_weight_ = loader.loadWeight(prefix + ".wv.weight");
    output_weight_ = loader.loadWeight(prefix + ".wo.weight");
}

mlx::core::array AttentionImplementation::forward(
    const mlx::core::array& hidden_states,
    const mlx::core::array& attention_mask,
    const std::pair<mlx::core::array, mlx::core::array>& kv_cache) {
    
    auto batch_size = hidden_states.shape()[0];
    auto seq_length = hidden_states.shape()[1];
    
    // Project hidden states to query, key, value
    auto query = mlx::core::matmul(hidden_states, query_weight_);
    auto key = mlx::core::matmul(hidden_states, key_weight_);
    auto value = mlx::core::matmul(hidden_states, value_weight_);
    
    // Reshape for multi-head attention
    query = mlx::core::reshape(query, {batch_size, seq_length, num_heads_, head_dim_});
    key = mlx::core::reshape(key, {batch_size, seq_length, num_heads_, head_dim_});
    value = mlx::core::reshape(value, {batch_size, seq_length, num_heads_, head_dim_});
    
    // Use MLX's built-in attention mechanism for now
    // In a real implementation, we would use a custom optimized version
    auto attn_output = mlx::nn::scaled_dot_product_attention(
        query, key, value, attention_mask, dropout_prob_);
    
    // Reshape back and project to output dimension
    attn_output = mlx::core::reshape(attn_output, {batch_size, seq_length, hidden_size_});
    attn_output = mlx::core::matmul(attn_output, output_weight_);
    
    return attn_output;
}

std::pair<mlx::core::array, mlx::core::array> AttentionImplementation::getKVCache() const {
    // In Phase 1, this is a simplified implementation
    // In reality, we would manage a more sophisticated cache
    return {key_cache_, value_cache_};
}

void AttentionImplementation::updateKVCache(const mlx::core::array& key, const mlx::core::array& value) {
    if (key_cache_.size() == 0) {
        key_cache_ = key;
        value_cache_ = value;
    } else {
        key_cache_ = mlx::core::concatenate({key_cache_, key}, 1);  // Concat along sequence dimension
        value_cache_ = mlx::core::concatenate({value_cache_, value}, 1);
    }
}

} // namespace mlx_transformer
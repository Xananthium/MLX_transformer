#include "transformer_block.h"

#include <mlx/nn/layers.h>

namespace mlx_transformer {

TransformerBlock::TransformerBlock(
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_attention_heads,
    float layer_norm_epsilon,
    float dropout_prob)
    : hidden_size_(hidden_size),
      layer_norm_epsilon_(layer_norm_epsilon) {
    
    // Initialize components
    attention_ = std::make_unique<AttentionImplementation>(
        hidden_size, num_attention_heads, dropout_prob);
    
    feed_forward_ = std::make_unique<FeedForward>(
        hidden_size, intermediate_size, dropout_prob);
    
    // Layer normalization parameters
    attention_ln_weight_ = mlx::core::ones({hidden_size_}, mlx::core::float32);
    attention_ln_bias_ = mlx::core::zeros({hidden_size_}, mlx::core::float32);
    
    ffn_ln_weight_ = mlx::core::ones({hidden_size_}, mlx::core::float32);
    ffn_ln_bias_ = mlx::core::zeros({hidden_size_}, mlx::core::float32);
}

void TransformerBlock::loadWeights(ModelLoader& loader, const std::string& prefix) {
    // Load attention weights
    attention_->loadWeights(loader, prefix + ".attention");
    
    // Load feed-forward weights
    feed_forward_->loadWeights(loader, prefix + ".mlp");
    
    // Load layer norm weights
    attention_ln_weight_ = loader.loadWeight(prefix + ".attention_norm.weight");
    // Some models might not have bias
    try {
        attention_ln_bias_ = loader.loadWeight(prefix + ".attention_norm.bias");
    } catch (...) {
        // Bias not present, use zeros
    }
    
    ffn_ln_weight_ = loader.loadWeight(prefix + ".mlp_norm.weight");
    try {
        ffn_ln_bias_ = loader.loadWeight(prefix + ".mlp_norm.bias");
    } catch (...) {
        // Bias not present, use zeros
    }
}

mlx::core::array TransformerBlock::forward(
    const mlx::core::array& hidden_states,
    const mlx::core::array& attention_mask) {
    
    // First sublayer: Self-attention with residual connection
    auto norm_input = mlx::nn::layer_norm(
        hidden_states, attention_ln_weight_, attention_ln_bias_, layer_norm_epsilon_);
    
    auto attn_output = attention_->forward(norm_input, attention_mask);
    auto residual = mlx::core::add(hidden_states, attn_output);
    
    // Second sublayer: Feed-forward network with residual connection
    auto ffn_norm_input = mlx::nn::layer_norm(
        residual, ffn_ln_weight_, ffn_ln_bias_, layer_norm_epsilon_);
    
    auto ffn_output = feed_forward_->forward(ffn_norm_input);
    auto output = mlx::core::add(residual, ffn_output);
    
    return output;
}

std::pair<mlx::core::array, mlx::core::array> TransformerBlock::getKVCache() const {
    return attention_->getKVCache();
}

void TransformerBlock::updateKVCache(const mlx::core::array& key, const mlx::core::array& value) {
    attention_->updateKVCache(key, value);
}

} // namespace mlx_transformer
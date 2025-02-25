#include "transformer_model.h"

#include <mlx/ops.h>
#include <mlx/nn/layers.h>
#include <mlx/random.h>

namespace mlx_transformer {

TransformerModel::TransformerModel(const ModelConfig& config)
    : config_(config) {
    
    // Initialize embedding layers
    token_embedding_ = mlx::core::zeros({config.vocab_size, config.hidden_size}, mlx::core::float32);
    
    // Initialize transformer layers
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; i++) {
        layers_.push_back(std::make_unique<TransformerBlock>(
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            config.layer_norm_epsilon));
    }
    
    // Initialize LM head (projection to vocabulary)
    lm_head_weight_ = mlx::core::zeros({config.hidden_size, config.vocab_size}, mlx::core::float32);
    
    // Initialize final layer norm
    final_ln_weight_ = mlx::core::ones({config.hidden_size}, mlx::core::float32);
    final_ln_bias_ = mlx::core::zeros({config.hidden_size}, mlx::core::float32);
}

void TransformerModel::loadWeights(ModelLoader& loader) {
    // Load embedding
    token_embedding_ = loader.loadWeight("embedding.weight");
    
    // Load transformer layers
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        layers_[i]->loadWeights(loader, "transformer.layers." + std::to_string(i));
    }
    
    // Load LM head
    lm_head_weight_ = loader.loadWeight("lm_head.weight");
    
    // Load final layer norm
    final_ln_weight_ = loader.loadWeight("transformer.ln_f.weight");
    try {
        final_ln_bias_ = loader.loadWeight("transformer.ln_f.bias");
    } catch (...) {
        // Bias not present, use zeros
    }
}

mlx::core::array TransformerModel::forward(
    const mlx::core::array& input_ids,
    const mlx::core::array& attention_mask) {
    
    // Get input embeddings
    auto hidden_states = mlx::core::take(token_embedding_, input_ids, 0);
    
    // Pass through transformer layers
    for (int i = 0; i < layers_.size(); i++) {
        hidden_states = layers_[i]->forward(hidden_states, attention_mask);
    }
    
    // Apply final layer norm
    hidden_states = mlx::nn::layer_norm(
        hidden_states, final_ln_weight_, final_ln_bias_, config_.layer_norm_epsilon);
    
    // Project to vocabulary
    auto logits = mlx::core::matmul(hidden_states, mlx::core::transpose(lm_head_weight_, {1, 0}));
    
    return logits;
}

mlx::core::array TransformerModel::generate_next_token(
    const mlx::core::array& input_ids,
    float temperature,
    int top_k) {
    
    // Forward pass to get logits for the last token
    auto logits = forward(input_ids);
    
    // Get logits for the last token in the sequence
    auto last_token_logits = mlx::core::take(
        logits,
        mlx::core::array({logits.shape()[1] - 1}),
        1);
    
    last_token_logits = mlx::core::squeeze(last_token_logits, 1);
    
    // Apply temperature
    if (temperature > 0) {
        last_token_logits = mlx::core::divide(last_token_logits, temperature);
    }
    
    // Apply top-k sampling if specified
    if (top_k > 0) {
        // Sort logits in descending order and keep only top-k
        auto sorted_indices = mlx::core::argsort(last_token_logits, -1, true);
        sorted_indices = mlx::core::slice(sorted_indices, {0}, {top_k});
        
        // Create a mask for top-k indices
        auto mask = mlx::core::zeros_like(last_token_logits);
        mask = mlx::core::scatter(mask, sorted_indices, mlx::core::ones({top_k}, mask.dtype()), 0);
        
        // Apply mask to logits (set non-top-k values to -inf)
        auto masked_logits = mlx::core::where(
            mask,
            last_token_logits,
            mlx::core::full_like(last_token_logits, -1e10));
        
        last_token_logits = masked_logits;
    }
    
    // Apply softmax to get probabilities
    auto probs = mlx::core::softmax(last_token_logits, -1);
    
    // Sample from the distribution
    auto next_token = mlx::core::random::categorical(probs);
    
    return next_token;
}

void TransformerModel::clearKVCache() {
    for (auto& layer : layers_) {
        layer->updateKVCache(mlx::core::array(), mlx::core::array());
    }
}

} // namespace mlx_transformer
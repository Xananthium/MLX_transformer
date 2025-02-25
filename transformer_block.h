#pragma once

#include <mlx/array.h>
#include <memory>
#include <utility>
#include <string>

#include "attention.h"
#include "feed_forward.h"
#include "model_loader.h"

namespace mlx_transformer {

class TransformerBlock {
public:
    TransformerBlock(
        int64_t hidden_size,
        int64_t intermediate_size,
        int64_t num_attention_heads,
        float layer_norm_epsilon = 1e-5,
        float dropout_prob = 0.0);
    
    void loadWeights(ModelLoader& loader, const std::string& prefix);
    
    mlx::core::array forward(
        const mlx::core::array& hidden_states,
        const mlx::core::array& attention_mask = {});
    
    // Get KV cache for this layer
    std::pair<mlx::core::array, mlx::core::array> getKVCache() const;
    
    // Update KV cache for this layer
    void updateKVCache(const mlx::core::array& key, const mlx::core::array& value);

private:
    int64_t hidden_size_;
    float layer_norm_epsilon_;
    
    std::unique_ptr<AttentionImplementation> attention_;
    std::unique_ptr<FeedForward> feed_forward_;
    
    mlx::core::array attention_ln_weight_;
    mlx::core::array attention_ln_bias_;
    mlx::core::array ffn_ln_weight_;
    mlx::core::array ffn_ln_bias_;
};

} // namespace mlx_transformer
#pragma once

#include <mlx/array.h>
#include <utility>
#include <string>

#include "model_loader.h"

namespace mlx_transformer {

class AttentionImplementation {
public:
    AttentionImplementation(
        int64_t hidden_size,
        int64_t num_heads,
        float dropout_prob = 0.0);
    
    void loadWeights(ModelLoader& loader, const std::string& prefix);
    
    mlx::core::array forward(
        const mlx::core::array& hidden_states,
        const mlx::core::array& attention_mask = {},
        const std::pair<mlx::core::array, mlx::core::array>& kv_cache = {});
    
    // Returns current KV cache
    std::pair<mlx::core::array, mlx::core::array> getKVCache() const;
    
    // Updates KV cache with new key and value tensors
    void updateKVCache(const mlx::core::array& key, const mlx::core::array& value);

private:
    int64_t hidden_size_;
    int64_t num_heads_;
    int64_t head_dim_;
    float dropout_prob_;
    float scale_;
    
    mlx::core::array query_weight_;
    mlx::core::array key_weight_;
    mlx::core::array value_weight_;
    mlx::core::array output_weight_;
    
    // Simple KV cache for now
    mlx::core::array key_cache_;
    mlx::core::array value_cache_;
};

} // namespace mlx_transformer
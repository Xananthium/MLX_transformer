#pragma once

#include <mlx/array.h>
#include <vector>
#include <memory>

#include "model_loader.h"
#include "transformer_block.h"

namespace mlx_transformer {

class TransformerModel {
public:
    TransformerModel(const ModelConfig& config);
    
    void loadWeights(ModelLoader& loader);
    
    mlx::core::array forward(
        const mlx::core::array& input_ids,
        const mlx::core::array& attention_mask = {});
    
    // Generate next token for sequence generation
    mlx::core::array generate_next_token(
        const mlx::core::array& input_ids,
        float temperature = 1.0,
        int top_k = 0);
    
    // Clear KV cache for all layers
    void clearKVCache();

private:
    ModelConfig config_;
    
    mlx::core::array token_embedding_;
    std::vector<std::unique_ptr<TransformerBlock>> layers_;
    mlx::core::array lm_head_weight_;
    mlx::core::array final_ln_weight_;
    mlx::core::array final_ln_bias_;
};

} // namespace mlx_transformer
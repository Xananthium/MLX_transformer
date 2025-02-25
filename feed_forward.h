#pragma once

#include <mlx/array.h>
#include <string>

#include "model_loader.h"

namespace mlx_transformer {

class FeedForward {
public:
    FeedForward(int64_t hidden_size, int64_t intermediate_size, float dropout_prob = 0.0);
    
    void loadWeights(ModelLoader& loader, const std::string& prefix);
    
    mlx::core::array forward(const mlx::core::array& hidden_states);

private:
    int64_t hidden_size_;
    int64_t intermediate_size_;
    float dropout_prob_;
    
    mlx::core::array gate_weight_;
    mlx::core::array up_weight_;
    mlx::core::array down_weight_;
};

} // namespace mlx_transformer